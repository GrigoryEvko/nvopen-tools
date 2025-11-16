// Function: sub_21CEF00
// Address: 0x21cef00
//
__int64 __fastcall sub_21CEF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  bool v22; // cc
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  char v27; // al

  switch ( a5 )
  {
    case 0xE4Du:
    case 0xE4Eu:
    case 0xE4Fu:
    case 0xE50u:
    case 0xE51u:
    case 0xE52u:
    case 0xE53u:
    case 0xE54u:
    case 0xE61u:
    case 0xE62u:
    case 0xE63u:
    case 0xE64u:
    case 0xE65u:
    case 0xE66u:
    case 0xE67u:
    case 0xE68u:
    case 0xE69u:
    case 0xE6Au:
    case 0xE6Bu:
    case 0xE6Cu:
    case 0xE6Du:
    case 0xE6Eu:
    case 0xE6Fu:
    case 0xE70u:
    case 0xE83u:
    case 0xE84u:
      v8 = sub_15F2050(a3);
      v9 = sub_1632FA0(v8);
      *(_DWORD *)a2 = 44;
      LOBYTE(v10) = sub_204D4D0(a1, v9, *(_QWORD *)a3);
      *(_QWORD *)(a2 + 16) = v11;
      *(_DWORD *)(a2 + 8) = v10;
      v12 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      *(_WORD *)(a2 + 44) = 3;
      *(_DWORD *)(a2 + 32) = 0;
      *(_QWORD *)(a2 + 24) = v12;
      *(_DWORD *)(a2 + 40) = 0;
      return 1;
    case 0xFDCu:
    case 0xFDDu:
    case 0xFDEu:
      v24 = sub_15F2050(a3);
      v25 = sub_1632FA0(v24);
      *(_DWORD *)a2 = 44;
      v17 = v25;
      if ( a5 == 4062 )
        goto LABEL_25;
      goto LABEL_20;
    case 0xFE5u:
    case 0xFE6u:
    case 0xFE7u:
      v15 = sub_15F2050(a3);
      v16 = sub_1632FA0(v15);
      *(_DWORD *)a2 = 44;
      v17 = v16;
      if ( a5 == 4071 )
      {
LABEL_25:
        v26 = 8 * sub_15A9520(v17, 0);
        if ( v26 == 32 )
        {
          v27 = 5;
        }
        else if ( v26 > 0x20 )
        {
          v27 = 6;
          if ( v26 != 64 )
          {
            v27 = 0;
            if ( v26 == 128 )
              v27 = 7;
          }
        }
        else
        {
          v27 = 3;
          if ( v26 != 8 )
            v27 = 4 * (v26 == 16);
        }
        *(_BYTE *)(a2 + 8) = v27;
        *(_QWORD *)(a2 + 16) = 0;
      }
      else
      {
LABEL_20:
        LOBYTE(v18) = sub_204D4D0(a1, v17, *(_QWORD *)a3);
        *(_DWORD *)(a2 + 8) = v18;
        *(_QWORD *)(a2 + 16) = v19;
      }
      v20 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      *(_WORD *)(a2 + 44) = 1;
      *(_DWORD *)(a2 + 32) = 0;
      *(_QWORD *)(a2 + 24) = v20;
      v21 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
      v22 = *(_DWORD *)(v21 + 32) <= 0x40u;
      v23 = *(_QWORD **)(v21 + 24);
      if ( !v22 )
        v23 = (_QWORD *)*v23;
      *(_DWORD *)(a2 + 40) = (_DWORD)v23;
      return 1;
    case 0x11A5u:
    case 0x11A6u:
    case 0x11A7u:
    case 0x11B1u:
    case 0x11B2u:
    case 0x11B3u:
    case 0x11BDu:
    case 0x11BEu:
    case 0x11BFu:
    case 0x11C6u:
    case 0x11C7u:
    case 0x11C8u:
    case 0x11D2u:
    case 0x11D3u:
    case 0x11D4u:
    case 0x11DEu:
    case 0x11DFu:
    case 0x11E0u:
    case 0x11E7u:
    case 0x11E8u:
    case 0x11E9u:
    case 0x11F3u:
    case 0x11F4u:
    case 0x11F5u:
    case 0x11FFu:
    case 0x1200u:
    case 0x1201u:
    case 0x1208u:
    case 0x1209u:
    case 0x120Au:
    case 0x1214u:
    case 0x1215u:
    case 0x1216u:
    case 0x1220u:
    case 0x1221u:
    case 0x1222u:
    case 0x1229u:
    case 0x122Au:
    case 0x122Bu:
    case 0x1235u:
    case 0x1236u:
    case 0x1237u:
    case 0x1241u:
    case 0x1242u:
    case 0x1243u:
    case 0x124Au:
    case 0x124Bu:
    case 0x124Cu:
    case 0x1256u:
    case 0x1257u:
    case 0x1258u:
    case 0x1262u:
    case 0x1263u:
    case 0x1264u:
      *(_BYTE *)(a2 + 8) = 4;
      *(_DWORD *)a2 = word_435D800[a5 - 4517];
      goto LABEL_4;
    case 0x11A8u:
    case 0x11A9u:
    case 0x11AAu:
    case 0x11B4u:
    case 0x11B5u:
    case 0x11B6u:
    case 0x11C0u:
    case 0x11C1u:
    case 0x11C2u:
    case 0x11C9u:
    case 0x11CAu:
    case 0x11CBu:
    case 0x11D5u:
    case 0x11D6u:
    case 0x11D7u:
    case 0x11E1u:
    case 0x11E2u:
    case 0x11E3u:
    case 0x11EAu:
    case 0x11EBu:
    case 0x11ECu:
    case 0x11F6u:
    case 0x11F7u:
    case 0x11F8u:
    case 0x1202u:
    case 0x1203u:
    case 0x1204u:
    case 0x120Bu:
    case 0x120Cu:
    case 0x120Du:
    case 0x1217u:
    case 0x1218u:
    case 0x1219u:
    case 0x1223u:
    case 0x1224u:
    case 0x1225u:
    case 0x122Cu:
    case 0x122Du:
    case 0x122Eu:
    case 0x1238u:
    case 0x1239u:
    case 0x123Au:
    case 0x1244u:
    case 0x1245u:
    case 0x1246u:
    case 0x124Du:
    case 0x124Eu:
    case 0x124Fu:
    case 0x1259u:
    case 0x125Au:
    case 0x125Bu:
    case 0x1265u:
    case 0x1266u:
    case 0x1267u:
      *(_BYTE *)(a2 + 8) = 5;
      *(_DWORD *)a2 = word_435D800[a5 - 4517];
      goto LABEL_4;
    case 0x11ABu:
    case 0x11ACu:
    case 0x11ADu:
    case 0x11B7u:
    case 0x11B8u:
    case 0x11B9u:
    case 0x11CCu:
    case 0x11CDu:
    case 0x11CEu:
    case 0x11D8u:
    case 0x11D9u:
    case 0x11DAu:
    case 0x11EDu:
    case 0x11EEu:
    case 0x11EFu:
    case 0x11F9u:
    case 0x11FAu:
    case 0x11FBu:
    case 0x120Eu:
    case 0x120Fu:
    case 0x1210u:
    case 0x121Au:
    case 0x121Bu:
    case 0x121Cu:
    case 0x122Fu:
    case 0x1230u:
    case 0x1231u:
    case 0x123Bu:
    case 0x123Cu:
    case 0x123Du:
    case 0x1250u:
    case 0x1251u:
    case 0x1252u:
    case 0x125Cu:
    case 0x125Du:
    case 0x125Eu:
      *(_BYTE *)(a2 + 8) = 6;
      *(_DWORD *)a2 = word_435D800[a5 - 4517];
      goto LABEL_4;
    case 0x11AEu:
    case 0x11AFu:
    case 0x11B0u:
    case 0x11BAu:
    case 0x11BBu:
    case 0x11BCu:
    case 0x11C3u:
    case 0x11C4u:
    case 0x11C5u:
    case 0x11CFu:
    case 0x11D0u:
    case 0x11D1u:
    case 0x11DBu:
    case 0x11DCu:
    case 0x11DDu:
    case 0x11E4u:
    case 0x11E5u:
    case 0x11E6u:
    case 0x11F0u:
    case 0x11F1u:
    case 0x11F2u:
    case 0x11FCu:
    case 0x11FDu:
    case 0x11FEu:
    case 0x1205u:
    case 0x1206u:
    case 0x1207u:
    case 0x1211u:
    case 0x1212u:
    case 0x1213u:
    case 0x121Du:
    case 0x121Eu:
    case 0x121Fu:
    case 0x1226u:
    case 0x1227u:
    case 0x1228u:
    case 0x1232u:
    case 0x1233u:
    case 0x1234u:
    case 0x123Eu:
    case 0x123Fu:
    case 0x1240u:
    case 0x1247u:
    case 0x1248u:
    case 0x1249u:
    case 0x1253u:
    case 0x1254u:
    case 0x1255u:
    case 0x125Fu:
    case 0x1260u:
    case 0x1261u:
    case 0x1268u:
    case 0x1269u:
    case 0x126Au:
      *(_BYTE *)(a2 + 8) = 3;
      *(_DWORD *)a2 = word_435D800[a5 - 4517];
      goto LABEL_4;
    case 0x1394u:
    case 0x1397u:
    case 0x139Au:
    case 0x139Bu:
    case 0x13A0u:
    case 0x13A3u:
    case 0x13A6u:
    case 0x13A7u:
    case 0x13ACu:
    case 0x13AFu:
    case 0x13B2u:
    case 0x13B3u:
    case 0x13B8u:
    case 0x13BBu:
    case 0x13BEu:
    case 0x13BFu:
    case 0x13C4u:
    case 0x13C7u:
    case 0x13CAu:
    case 0x13CBu:
    case 0x13D0u:
    case 0x13D3u:
    case 0x13D6u:
    case 0x13D9u:
    case 0x1422u:
    case 0x1425u:
    case 0x1428u:
    case 0x1429u:
    case 0x142Eu:
    case 0x1431u:
    case 0x1434u:
    case 0x1435u:
    case 0x143Au:
    case 0x143Du:
    case 0x1440u:
    case 0x1441u:
    case 0x1446u:
    case 0x1449u:
    case 0x144Cu:
    case 0x144Du:
    case 0x1452u:
    case 0x1455u:
    case 0x1458u:
    case 0x1459u:
    case 0x145Eu:
    case 0x1461u:
    case 0x1464u:
    case 0x1467u:
    case 0x146Au:
    case 0x146Du:
    case 0x1472u:
    case 0x1475u:
    case 0x1478u:
    case 0x147Bu:
    case 0x147Eu:
    case 0x1481u:
    case 0x1484u:
    case 0x1487u:
      *(_BYTE *)(a2 + 8) = 91;
      *(_DWORD *)a2 = word_435D9A0[a5 - 5012];
      goto LABEL_4;
    case 0x1395u:
    case 0x1396u:
    case 0x1398u:
    case 0x1399u:
    case 0x139Cu:
    case 0x139Du:
    case 0x139Eu:
    case 0x139Fu:
    case 0x13A1u:
    case 0x13A2u:
    case 0x13A4u:
    case 0x13A5u:
    case 0x13A8u:
    case 0x13A9u:
    case 0x13AAu:
    case 0x13ABu:
    case 0x13ADu:
    case 0x13AEu:
    case 0x13B0u:
    case 0x13B1u:
    case 0x13B4u:
    case 0x13B5u:
    case 0x13B6u:
    case 0x13B7u:
    case 0x13B9u:
    case 0x13BAu:
    case 0x13BCu:
    case 0x13BDu:
    case 0x13C0u:
    case 0x13C1u:
    case 0x13C2u:
    case 0x13C3u:
    case 0x13C5u:
    case 0x13C6u:
    case 0x13C8u:
    case 0x13C9u:
    case 0x13CCu:
    case 0x13CDu:
    case 0x13CEu:
    case 0x13CFu:
    case 0x13D1u:
    case 0x13D2u:
    case 0x13D4u:
    case 0x13D5u:
    case 0x13D7u:
    case 0x13D8u:
    case 0x13DAu:
    case 0x13DBu:
    case 0x1423u:
    case 0x1424u:
    case 0x1426u:
    case 0x1427u:
    case 0x142Au:
    case 0x142Bu:
    case 0x142Cu:
    case 0x142Du:
    case 0x142Fu:
    case 0x1430u:
    case 0x1432u:
    case 0x1433u:
    case 0x1436u:
    case 0x1437u:
    case 0x1438u:
    case 0x1439u:
    case 0x143Bu:
    case 0x143Cu:
    case 0x143Eu:
    case 0x143Fu:
    case 0x1442u:
    case 0x1443u:
    case 0x1444u:
    case 0x1445u:
    case 0x1447u:
    case 0x1448u:
    case 0x144Au:
    case 0x144Bu:
    case 0x144Eu:
    case 0x144Fu:
    case 0x1450u:
    case 0x1451u:
    case 0x1453u:
    case 0x1454u:
    case 0x1456u:
    case 0x1457u:
    case 0x145Au:
    case 0x145Bu:
    case 0x145Cu:
    case 0x145Du:
    case 0x145Fu:
    case 0x1460u:
    case 0x1462u:
    case 0x1463u:
    case 0x1465u:
    case 0x1466u:
    case 0x1468u:
    case 0x1469u:
    case 0x146Bu:
    case 0x146Cu:
    case 0x146Eu:
    case 0x146Fu:
    case 0x1473u:
    case 0x1474u:
    case 0x1476u:
    case 0x1477u:
    case 0x1479u:
    case 0x147Au:
    case 0x147Cu:
    case 0x147Du:
    case 0x147Fu:
    case 0x1480u:
    case 0x1482u:
    case 0x1483u:
    case 0x1485u:
    case 0x1486u:
    case 0x1488u:
    case 0x1489u:
      *(_BYTE *)(a2 + 8) = 43;
      *(_DWORD *)a2 = word_435D9A0[a5 - 5012];
LABEL_4:
      *(_QWORD *)(a2 + 16) = 0;
      *(_QWORD *)(a2 + 24) = 0;
      *(_DWORD *)(a2 + 32) = 0;
      *(_WORD *)(a2 + 44) = 1;
      *(_DWORD *)(a2 + 40) = 16;
      return 1;
    case 0x1548u:
    case 0x1549u:
    case 0x154Au:
    case 0x154Bu:
    case 0x154Cu:
    case 0x154Du:
    case 0x154Eu:
    case 0x154Fu:
    case 0x1580u:
    case 0x1581u:
    case 0x1582u:
    case 0x1583u:
    case 0x1584u:
    case 0x1585u:
    case 0x1586u:
    case 0x1587u:
    case 0x15B8u:
    case 0x15B9u:
    case 0x15BAu:
    case 0x15BBu:
    case 0x15BCu:
    case 0x15BDu:
    case 0x15BEu:
    case 0x15BFu:
      *(_DWORD *)a2 = 44;
      *(_BYTE *)(a2 + 8) = 88;
      goto LABEL_13;
    case 0x1550u:
    case 0x1552u:
    case 0x1554u:
    case 0x1556u:
    case 0x1588u:
    case 0x158Au:
    case 0x158Cu:
    case 0x158Eu:
    case 0x15C0u:
    case 0x15C2u:
    case 0x15C4u:
    case 0x15C6u:
      *(_DWORD *)a2 = 44;
      *(_BYTE *)(a2 + 8) = 87;
      goto LABEL_13;
    case 0x1551u:
    case 0x1553u:
    case 0x1555u:
    case 0x1557u:
    case 0x1589u:
    case 0x158Bu:
    case 0x158Du:
    case 0x158Fu:
    case 0x15C1u:
    case 0x15C3u:
    case 0x15C5u:
    case 0x15C7u:
      *(_DWORD *)a2 = 44;
      *(_BYTE *)(a2 + 8) = 92;
LABEL_13:
      *(_QWORD *)(a2 + 16) = 0;
      v13 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      *(_WORD *)(a2 + 44) = 1;
      *(_DWORD *)(a2 + 32) = 0;
      *(_QWORD *)(a2 + 24) = v13;
      *(_DWORD *)(a2 + 40) = 16;
      return 1;
    case 0x1578u:
    case 0x157Au:
    case 0x157Cu:
    case 0x157Eu:
    case 0x15B0u:
    case 0x15B2u:
    case 0x15B4u:
    case 0x15B6u:
    case 0x15E8u:
    case 0x15EAu:
    case 0x15ECu:
    case 0x15EEu:
      *(_DWORD *)a2 = 45;
      *(_BYTE *)(a2 + 8) = 87;
      goto LABEL_17;
    case 0x1579u:
    case 0x157Bu:
    case 0x157Du:
    case 0x157Fu:
    case 0x15B1u:
    case 0x15B3u:
    case 0x15B5u:
    case 0x15B7u:
    case 0x15E9u:
    case 0x15EBu:
    case 0x15EDu:
    case 0x15EFu:
      *(_DWORD *)a2 = 45;
      *(_BYTE *)(a2 + 8) = 92;
LABEL_17:
      *(_QWORD *)(a2 + 16) = 0;
      v14 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
      *(_WORD *)(a2 + 44) = 2;
      *(_DWORD *)(a2 + 32) = 0;
      *(_QWORD *)(a2 + 24) = v14;
      *(_DWORD *)(a2 + 40) = 16;
      result = 1;
      break;
    default:
      result = sub_2176FC0(a2, a3, a5);
      break;
  }
  return result;
}
