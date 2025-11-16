// Function: sub_D5D1D0
// Address: 0xd5d1d0
//
__int64 __fastcall sub_D5D1D0(unsigned __int8 *a1, __int64 *a2, __int64 **a3)
{
  int v3; // eax
  unsigned __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v9; // rax
  __int64 v10; // rsi
  char v11; // al
  __m128i v12; // [rsp-48h] [rbp-48h] BYREF
  char v13; // [rsp-30h] [rbp-30h]

  v3 = *a1;
  if ( (unsigned __int8)v3 <= 0x1Cu )
    return 0;
  v5 = (unsigned int)(v3 - 34);
  if ( (unsigned __int8)v5 > 0x33u )
    return 0;
  v7 = 0x8000000000041LL;
  if ( _bittest64(&v7, v5) )
  {
    v9 = sub_D5BAA0(a1);
    v10 = v9;
    if ( v9 )
    {
      if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v9 + 24) + 16LL) + 8LL) == 14 )
      {
        sub_D5BC90(&v12, v9, 3u, a2);
        if ( v13 )
          return sub_ACA8A0(a3);
      }
    }
    v11 = sub_D5BB80(a1);
    if ( (v11 & 8) != 0 )
      return sub_ACA8A0(a3);
    if ( (v11 & 0x10) != 0 )
      return sub_AD6530((__int64)a3, v10);
  }
  return 0;
}
