// Function: sub_BAC2C0
// Address: 0xbac2c0
//
__int64 __fastcall sub_BAC2C0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rcx
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v9; // [rsp+18h] [rbp-78h]
  __m128i *v10; // [rsp+20h] [rbp-70h] BYREF
  __int64 v11; // [rsp+28h] [rbp-68h]
  __m128i v12; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+40h] [rbp-50h] BYREF
  __int64 v14; // [rsp+48h] [rbp-48h]
  _QWORD v15[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = *a2;
  v9 = a2[1];
  if ( *a2 == v9 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
  }
  else
  {
    v11 = 1;
    v3 = v2;
    v10 = &v12;
    v12.m128i_i16[0] = 91;
    do
    {
      v13 = v15;
      sub_BABE50((__int64 *)&v13, *(_BYTE **)v3, *(_QWORD *)v3 + *(_QWORD *)(v3 + 8));
      if ( v14 == 0x3FFFFFFFFFFFFFFFLL )
        goto LABEL_14;
      sub_2241490(&v13, ",", 1, v4);
      sub_2241490(&v10, v13, v14, v5);
      if ( v13 != v15 )
        j_j___libc_free_0(v13, v15[0] + 1LL);
      v3 += 32;
    }
    while ( v9 != v3 );
    sub_2240CE0(&v10, v11 - 1, 1);
    if ( v11 == 0x3FFFFFFFFFFFFFFFLL || v11 == 4611686018427387902LL )
LABEL_14:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(&v10, "];", 2, v6);
    sub_2241490(&v10, a2[3], a2[4], v7);
    *(_QWORD *)a1 = a1 + 16;
    if ( v10 == &v12 )
    {
      *(__m128i *)(a1 + 16) = _mm_load_si128(&v12);
    }
    else
    {
      *(_QWORD *)a1 = v10;
      *(_QWORD *)(a1 + 16) = v12.m128i_i64[0];
    }
    *(_QWORD *)(a1 + 8) = v11;
  }
  return a1;
}
