// Function: sub_36D7360
// Address: 0x36d7360
//
__int64 __fastcall sub_36D7360(unsigned int a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  bool v5; // al
  unsigned int v6; // eax

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 365 )
  {
    if ( v4 > 470 )
    {
      if ( v4 == 497 )
        goto LABEL_10;
    }
    else if ( v4 > 464 )
    {
      goto LABEL_10;
    }
    goto LABEL_5;
  }
  if ( v4 > 337 )
    goto LABEL_10;
  if ( v4 > 294 )
  {
    if ( (unsigned int)(v4 - 298) <= 1 )
      goto LABEL_10;
LABEL_5:
    v5 = 0;
    if ( (*(_BYTE *)(a2 + 32) & 2) == 0 )
      goto LABEL_6;
    goto LABEL_10;
  }
  if ( v4 <= 292 )
    goto LABEL_5;
LABEL_10:
  v6 = sub_2EAC1E0(*(_QWORD *)(a2 + 112));
  v5 = sub_AE2980(a3, v6)[1] == 64;
LABEL_6:
  if ( a1 > 0x22D2 )
    goto LABEL_71;
  if ( a1 > 0x227C )
  {
    switch ( a1 )
    {
      case 0x227Du:
      case 0x227Eu:
        return (unsigned int)v5 + 1516;
      case 0x227Fu:
        return (unsigned int)v5 + 1518;
      case 0x2280u:
        return (unsigned int)v5 + 1520;
      case 0x2281u:
        return 1522;
      case 0x2282u:
        return 1523;
      case 0x2283u:
        return 1524;
      case 0x2284u:
        return 1525;
      case 0x2285u:
        return (unsigned int)v5 + 1526;
      case 0x2286u:
        return (unsigned int)v5 + 1528;
      case 0x228Bu:
      case 0x228Cu:
        return (unsigned int)v5 + 1530;
      case 0x228Du:
        return (unsigned int)v5 + 1532;
      case 0x228Eu:
        return (unsigned int)v5 + 1534;
      case 0x228Fu:
        return 1536;
      case 0x2290u:
        return 1537;
      case 0x2291u:
        return 1538;
      case 0x2292u:
        return 1539;
      case 0x2293u:
        return (unsigned int)v5 + 1540;
      case 0x2294u:
        return (unsigned int)v5 + 1542;
      case 0x2295u:
      case 0x2296u:
        return (unsigned int)v5 + 1544;
      case 0x2297u:
        return (unsigned int)v5 + 1546;
      case 0x2298u:
        return (unsigned int)v5 + 1548;
      case 0x2299u:
        return 1550;
      case 0x229Au:
        return 1551;
      case 0x229Bu:
        return 1552;
      case 0x229Cu:
        return 1553;
      case 0x229Du:
        return (unsigned int)v5 + 1554;
      case 0x229Eu:
        return (unsigned int)v5 + 1556;
      case 0x22B3u:
      case 0x22B4u:
      case 0x22B5u:
      case 0x22B6u:
        return (unsigned int)v5 + 1566;
      case 0x22B7u:
        return (unsigned int)v5 + 1568;
      case 0x22B8u:
      case 0x22B9u:
        return 1570;
      case 0x22BAu:
        return (unsigned int)v5 + 1571;
      case 0x22BBu:
      case 0x22BCu:
        return (unsigned int)v5 + 1573;
      case 0x22BDu:
      case 0x22BEu:
        return (unsigned int)v5 + 1575;
      case 0x22BFu:
        return (unsigned int)v5 + 1577;
      case 0x22C0u:
      case 0x22C1u:
        return 1579;
      case 0x22C2u:
        return (unsigned int)v5 + 1580;
      case 0x22C3u:
      case 0x22C4u:
        return (unsigned int)v5 + 1582;
      case 0x22C5u:
      case 0x22C6u:
        return (unsigned int)v5 + 1584;
      case 0x22C7u:
        return (unsigned int)v5 + 1586;
      case 0x22C8u:
      case 0x22C9u:
        return 1588;
      case 0x22CAu:
        return (unsigned int)v5 + 1589;
      case 0x22CBu:
      case 0x22CCu:
      case 0x22CDu:
      case 0x22CEu:
        return (unsigned int)v5 + 1591;
      case 0x22CFu:
        return (unsigned int)v5 + 1593;
      case 0x22D0u:
      case 0x22D1u:
        return 1595;
      case 0x22D2u:
        return (unsigned int)v5 + 1596;
      default:
        goto LABEL_71;
    }
  }
  if ( a1 == 8279 )
    return 386;
  if ( a1 <= 0x2057 )
  {
    if ( a1 > 0x2055 )
      return (unsigned int)v5 + 384;
    if ( a1 > 0x2053 )
      return (unsigned int)v5 + 382;
LABEL_71:
    BUG();
  }
  if ( a1 != 8280 )
    goto LABEL_71;
  return (unsigned int)v5 + 387;
}
