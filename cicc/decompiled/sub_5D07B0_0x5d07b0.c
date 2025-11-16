// Function: sub_5D07B0
// Address: 0x5d07b0
//
__int64 __fastcall sub_5D07B0(__int64 a1, __int64 a2, char a3)
{
  char v4; // dl
  __int64 v5; // rbx
  char v6; // al
  char v7; // al
  __int64 i; // rdi

  if ( a3 == 3 )
  {
    if ( !(unsigned int)sub_8D3B10(*(_QWORD *)(a2 + 8)) )
    {
      sub_685360(1107, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
    if ( !(unsigned int)sub_8D23B0(*(_QWORD *)(a2 + 8)) )
    {
      for ( i = *(_QWORD *)(a2 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (unsigned int)sub_5D0610(i, a1 + 56) )
        *(_BYTE *)(a2 + 34) |= 4u;
    }
    return a2;
  }
  if ( a3 != 6 )
    sub_721090(a1);
  v4 = *(_BYTE *)(a2 + 140);
  if ( v4 == 12 )
  {
    v5 = a2;
    do
    {
      v5 = *(_QWORD *)(v5 + 160);
      v6 = *(_BYTE *)(v5 + 140);
    }
    while ( v6 == 12 );
  }
  else
  {
    v6 = *(_BYTE *)(a2 + 140);
    v5 = a2;
  }
  if ( v6 != 11 )
  {
    sub_685360(1107, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  v7 = *(_BYTE *)(a1 + 10);
  if ( (unsigned __int8)(v7 - 2) > 1u )
  {
    if ( v7 != 1 && (v7 != 6 || v4 != 12 || !*(_QWORD *)(a2 + 8) || (unsigned int)sub_8D23B0(v5)) )
    {
      sub_684B30(1108, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
    if ( !(unsigned int)sub_5D0610(v5, a1 + 56) )
    {
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
  }
  *(_BYTE *)(v5 + 179) |= 0x10u;
  return a2;
}
