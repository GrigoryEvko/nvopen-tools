// Function: sub_117CA00
// Address: 0x117ca00
//
bool __fastcall sub_117CA00(const __m128i *a1, __int64 *a2, char a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned int v6; // eax
  unsigned __int64 v7; // rdx
  __int16 v8; // ax
  __int16 v9; // bx
  bool result; // al
  __int16 v11; // ax
  unsigned __int64 v12; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-88h]
  __m128i v14[2]; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v15; // [rsp+30h] [rbp-60h]
  __int64 v16; // [rsp+38h] [rbp-58h]
  __m128i v17; // [rsp+40h] [rbp-50h]
  __int64 v18; // [rsp+50h] [rbp-40h]

  v18 = a1[10].m128i_i64[0];
  v5 = a2[1];
  v15 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v14[0] = _mm_loadu_si128(a1 + 6);
  v16 = a4;
  v14[1] = _mm_loadu_si128(a1 + 7);
  v17 = _mm_loadu_si128(a1 + 9);
  if ( *(_BYTE *)(v5 + 8) == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    v13 = v6;
    if ( v6 > 0x40 )
    {
      sub_C43690((__int64)&v12, -1, 1);
    }
    else
    {
      v7 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
      if ( !v6 )
        v7 = 0;
      v12 = v7;
    }
  }
  else
  {
    v13 = 1;
    v12 = 1;
  }
  if ( (a3 & 4) != 0 )
  {
    v8 = sub_9B3E70(a2, (__int64 *)&v12, 56, 0, v14);
    if ( (a3 & 2) != 0 )
      v8 &= 0x3FCu;
    v9 = v8 & 0x1FB;
  }
  else
  {
    v11 = sub_9B3E70(a2, (__int64 *)&v12, 60, 0, v14);
    v9 = v11 & 0x3FC;
    if ( (a3 & 2) == 0 )
      v9 = v11;
  }
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  result = 0;
  if ( (v9 & 0x207) == 0 )
  {
    result = 1;
    if ( (a3 & 8) == 0 )
      return (v9 & 0x3C) == 0;
  }
  return result;
}
