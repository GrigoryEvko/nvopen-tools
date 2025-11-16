// Function: sub_16D2060
// Address: 0x16d2060
//
_QWORD *__fastcall sub_16D2060(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 i; // rdx
  char v9; // al
  _BYTE *v10; // rcx

  v6 = a2[1];
  *a1 = a1 + 2;
  sub_2240A50(a1, v6, 0, a4, a5);
  v7 = a2[1];
  if ( v7 )
  {
    for ( i = 0; i != v7; ++i )
    {
      v9 = *(_BYTE *)(*a2 + i);
      v10 = (_BYTE *)(i + *a1);
      if ( (unsigned __int8)(v9 - 65) < 0x1Au )
        v9 += 32;
      *v10 = v9;
    }
  }
  return a1;
}
