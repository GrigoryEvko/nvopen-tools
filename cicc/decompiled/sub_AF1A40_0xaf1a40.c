// Function: sub_AF1A40
// Address: 0xaf1a40
//
__int64 __fastcall sub_AF1A40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v4; // [rsp+10h] [rbp-30h] BYREF
  __int64 v5; // [rsp+18h] [rbp-28h]
  unsigned int v6; // [rsp+20h] [rbp-20h]
  int v7; // [rsp+24h] [rbp-1Ch]

  v4 = a1;
  v5 = a2;
  v7 = 0;
  v3 = 0;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagZero", 10);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 13
      && *(_QWORD *)v4 == 0x725067616C464944LL
      && *(_DWORD *)(v4 + 8) == 1952544361
      && *(_BYTE *)(v4 + 12) == 101 )
    {
      v6 = 1;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 15
           && *(_QWORD *)v4 == 0x725067616C464944LL
           && *(_DWORD *)(v4 + 8) == 1667593327
           && *(_WORD *)(v4 + 12) == 25972
           && *(_BYTE *)(v4 + 14) == 100 )
    {
      v6 = 2;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 12 && *(_QWORD *)v4 == 0x755067616C464944LL && *(_DWORD *)(v4 + 8) == 1667853410 )
    {
      v6 = 3;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 4;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagFwdDecl", 13);
  v3 = 8;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagAppleBlock", 16);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 18
      && !(*(_QWORD *)v4 ^ 0x655267616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6942646576726573LL)
      && *(_WORD *)(v4 + 16) == 13428 )
    {
      v6 = 16;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 13
           && *(_QWORD *)v4 == 0x695667616C464944LL
           && *(_DWORD *)(v4 + 8) == 1635087474
           && *(_BYTE *)(v4 + 12) == 108 )
    {
      v6 = 32;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 64;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagArtificial", 16);
  if ( !(_BYTE)v7
    && v5 == 14
    && *(_QWORD *)v4 == 0x784567616C464944LL
    && *(_DWORD *)(v4 + 8) == 1667853424
    && *(_WORD *)(v4 + 12) == 29801 )
  {
    v6 = 128;
    LOBYTE(v7) = 1;
  }
  v3 = 256;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagPrototyped", 16);
  v3 = 512;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagObjcClassComplete", 23);
  if ( !(_BYTE)v7
    && v5 == 19
    && !(*(_QWORD *)v4 ^ 0x624F67616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6E696F507463656ALL)
    && *(_WORD *)(v4 + 16) == 25972
    && *(_BYTE *)(v4 + 18) == 114 )
  {
    v6 = 1024;
    LOBYTE(v7) = 1;
  }
  v3 = 2048;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagVector", 12);
  v3 = 4096;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagStaticMember", 18);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 21
      && !(*(_QWORD *)v4 ^ 0x564C67616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6566655265756C61LL)
      && *(_DWORD *)(v4 + 16) == 1668179314
      && *(_BYTE *)(v4 + 20) == 101 )
    {
      v6 = 0x2000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 21
           && !(*(_QWORD *)v4 ^ 0x565267616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6566655265756C61LL)
           && *(_DWORD *)(v4 + 16) == 1668179314
           && *(_BYTE *)(v4 + 20) == 101 )
    {
      v6 = 0x4000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 19
           && !(*(_QWORD *)v4 ^ 0x784567616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x626D795374726F70LL)
           && *(_WORD *)(v4 + 16) == 27759
           && *(_BYTE *)(v4 + 18) == 115 )
    {
      v6 = 0x8000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 23
           && !(*(_QWORD *)v4 ^ 0x695367616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x65686E49656C676ELL)
           && *(_DWORD *)(v4 + 16) == 1635019122
           && *(_WORD *)(v4 + 20) == 25454
           && *(_BYTE *)(v4 + 22) == 101 )
    {
      v6 = 0x10000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 25
           && !(*(_QWORD *)v4 ^ 0x754D67616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6E49656C7069746CLL)
           && *(_QWORD *)(v4 + 16) == 0x636E617469726568LL
           && *(_BYTE *)(v4 + 24) == 101 )
    {
      v6 = 0x20000;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 196608;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagVirtualInheritance", 24);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 23
      && !(*(_QWORD *)v4 ^ 0x6E4967616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x64656375646F7274LL)
      && *(_DWORD *)(v4 + 16) == 1953655126
      && *(_WORD *)(v4 + 20) == 24949
      && *(_BYTE *)(v4 + 22) == 108 )
    {
      v6 = 0x40000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 14
           && *(_QWORD *)v4 == 0x694267616C464944LL
           && *(_DWORD *)(v4 + 8) == 1701398132
           && *(_WORD *)(v4 + 12) == 25708 )
    {
      v6 = 0x80000;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 0x100000;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagNoReturn", 14);
  v3 = (int)&dword_400000;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagTypePassByValue", 21);
  if ( !(_BYTE)v7 )
  {
    if ( v5 == 25
      && !(*(_QWORD *)v4 ^ 0x795467616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x7942737361506570LL)
      && *(_QWORD *)(v4 + 16) == 0x636E657265666552LL
      && *(_BYTE *)(v4 + 24) == 101 )
    {
      v6 = 0x800000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 15
           && *(_QWORD *)v4 == 0x6E4567616C464944LL
           && *(_DWORD *)(v4 + 8) == 1816358261
           && *(_WORD *)(v4 + 12) == 29537
           && *(_BYTE *)(v4 + 14) == 115 )
    {
      v6 = (unsigned int)&loc_1000000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 11
           && *(_QWORD *)v4 == 0x685467616C464944LL
           && *(_WORD *)(v4 + 8) == 28277
           && *(_BYTE *)(v4 + 10) == 107 )
    {
      v6 = 0x2000000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 16 && !(*(_QWORD *)v4 ^ 0x6F4E67616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6C6169766972546ELL) )
    {
      v6 = 0x4000000;
      LOBYTE(v7) = 1;
    }
    else if ( v5 == 15
           && *(_QWORD *)v4 == 0x694267616C464944LL
           && *(_DWORD *)(v4 + 8) == 1684948327
           && *(_WORD *)(v4 + 12) == 24937
           && *(_BYTE *)(v4 + 14) == 110 )
    {
      v6 = 0x8000000;
      LOBYTE(v7) = 1;
    }
  }
  v3 = 0x10000000;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagLittleEndian", 18);
  if ( !(_BYTE)v7
    && v5 == 23
    && !(*(_QWORD *)v4 ^ 0x6C4167616C464944LL | *(_QWORD *)(v4 + 8) ^ 0x6544736C6C61436CLL)
    && *(_DWORD *)(v4 + 16) == 1769104243
    && *(_WORD *)(v4 + 20) == 25954
    && *(_BYTE *)(v4 + 22) == 100 )
  {
    v6 = 0x20000000;
    LOBYTE(v7) = 1;
  }
  v3 = 36;
  sub_AF12D0((__int64)&v4, &v3, "DIFlagIndirectVirtualBase", 25);
  result = 0;
  if ( (_BYTE)v7 )
    return v6;
  return result;
}
