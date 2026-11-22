# Failure examples of the quantized BERTurk (INT8)

## Missed offensive messages (false negatives), most confident first

* P(off)=0.01 — O günler geçti canımıniçi, bir daha karşılaşmamak dileğiyle
* P(off)=0.02 — Tanışmak isteyen ve sohbet etmek isteyen var mı ?   İyi geceler
* P(off)=0.02 — Sipammmmm yaa    Geldi hem laptopu aldı    Hem de beni esir aldı 😂😂
* P(off)=0.02 — Ay ben yine samimiyetsiz saatlerime geldim en iyisi yatmak
* P(off)=0.02 — oha yuh cus kac gundur sizi bensiz biraktigim icin sorryy😭😭😭😭
* P(off)=0.02 — anlık seni istemiyorum isimli wp kitabı okuyorum kusmama ramak kaldı
* P(off)=0.02 — @USER @USER merhaba efenfim.   orası doğru. yalnız der spigel "das zeitalter der autokraten" derken, reisicumhur "dünyayı şekillendiren liderler" yazıldı sanıyor! (:
* P(off)=0.03 — Değil unutmak acısı hafifleyenin kanı kurusun. #FıratYılmazÇakıroğlu
* P(off)=0.03 — @USER olgun kadın başka olur. Tercihim olgun kadınlar.
* P(off)=0.03 — @USER Yeeni yıla bak anasını satıyım6 Günündeeyiz 5 günü tatildiGeeldee seevmee böylee yılıafeerin yeeni yılafeerin

## Clean messages flagged as offensive (false positives)

* P(off)=1.00 — 1. Sınıftayken 4 hafta devamsızlığı burda bitiremem diyordum aqsdf yüzümü okulda yıkamaya başladım aqsdf
* P(off)=1.00 — Çok dertlisin aqam noldu sana?   Ağlayıp, o güldüğü fotoğraflarına bakıyordum yine)
* P(off)=1.00 — dalgasına f*nansal k*lelerle ilgili tweet atmıştım 30 dk içinde 20 kişi 5 senelik okul ücretimi karşılayacak şeyler yazmışlardı gram şaşırmadım cidden
* P(off)=0.99 — Gerizekalılar buna emin olun sizden çok Galatasaray taraftarı tepki gösterir. Ayrıca ofsaytı orta hakem vermiyor
* P(off)=0.98 — @USER @USER Öğrenmemeleri için öyle bir hale getirmişler ki bir bakın isterseniz.. okuyabilirseniz alkışlarız. Rezalet iprenç ve aşağılık şekilde çocuklarımızın zekasıyla oynanıyor😡
* P(off)=0.98 — @USER @USER Fetöye en başta karşı çıkan cemaat. İçeriye sokmayın bunları diye ikitdar partisine karşı çıkan hoca dediğiniz o şarlatan amerikanın adamı diye bas bas bağıran ama bi türlü kendini duyuramayan tek cemaattir.  Süleymanlı cemaati. Sizde bu iftiralarla çok güzel yere gelrsnz
* P(off)=0.98 — @USER Yaramaz çocuk gibiler gözünü ayırma ayarı yok ya la bunların.
* P(off)=0.97 — Az önce ablam bişey sordu he dedim yeğenim (2) he yok evet var edepsiz dedi folikasit çocuğu seni
* P(off)=0.97 — Blackpink'in bu kadar güzel fotoğrafı olmasaydı keşke ikide bir icon değiştirip duruyorum aw çok zor
* P(off)=0.97 — TV de   Erdoğan'a soruyorlar   İstanbul depreme hazır mı?   Cevap:Büyük depremde faturası ne olur belli değil.Ben bundan Endişeliyim korkuyorum      Bunu 25 senedir İstanbul'u,   17 senedir Türkiye'yi yöneten Cumhurbaşkanı söylüyor.      Allah belanı versin Lidyalılar ne hale getirdin İstanbul'u

## Lexicon baseline wrong, BERT right (where context helps)

* gold=NOT — @USER @USER Düşük satsa sahip hacimli projelerin hepsinde  farazi trend çizgilerinin hacimli kırılışı takip edilmeli. HOT yakın dönemde bunu yaptı ve ciddi bir momentum kazandı. Bu kırılım sonrası nereye gideceği tahmini ise tamamen kehanet olur. Tavsiyem bir sonraki kırılımı beklemeniz.
* gold=OFF — @USER Ama alırken annem odaya girdi hay aq
* gold=OFF — @USER Gidin bi kurban kesin , hamama gidin.. bişey yapın. bu ne cenabetliktir arkadaş.
* gold=NOT — Memem olsa sürekli dekolte giyer, her hareketimde de frikik verirdim
* gold=OFF — Cok basit bir numaraydı bu, beni mi hafife aldın yoksa zekan mı elvermedi
* gold=NOT — @USER Astronomi ile astroloji karıştırıp duruyoruz biri bilim dalı biri eğlence için yapılmış bir şey    Burçlar değil kişiliğinizi belirleyen şeyler    Hayat için de beslendiğiniz tüm etken çevre faktörler      Not bu arada ben başak ;-)
* gold=NOT — "İnsaf ile hakikate şahitlik yaparak Allah'a bağlılığınızda sıkı durun ve herhangi bir kimseye karşı nefretiniz, sizi adaletten sapma günahına itmesin." (Maide 5/8)
* gold=NOT — eve geldim makyajımı silmeye erindim az önce yüzümü yıkadım makyajımı temizledim içeri geçtim babam bağırarak diyo ki allahım bi insanın makyajsız hali bu kadarmi güzel olur yaaaaav xd krem sürmemi bile istemiyo yüzüme
* gold=NOT — @USER Oyumu Recep Tayyip Erdoğan 'a verdim, operasyondan onun haberi olduğuna inanmıyorum dediğini hepimiz duyduk, hiç sıkıntı yapma sen adres belli 🤣
* gold=NOT — @USER Aynen 1 saat önce falan gök delindi resmen nasıl yağdıysa kimse Temmuz ayında olduğumuza inanmaz
