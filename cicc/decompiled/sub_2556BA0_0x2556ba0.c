// Function: sub_2556BA0
// Address: 0x2556ba0
//
__m128i *__fastcall sub_2556BA0(__m128i *a1, __int64 a2)
{
  unsigned __int64 v4; // r15
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  char v7; // cl
  unsigned __int64 v8; // rdx
  unsigned int v9; // edi
  unsigned __int64 v10; // rsi
  unsigned int v11; // eax
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  char v15; // cl
  unsigned __int64 v16; // rdx
  unsigned int v17; // edi
  unsigned __int64 v18; // rsi
  unsigned int v19; // eax
  __m128i *v20; // rax
  unsigned __int64 v21; // rcx
  _BYTE *v23; // [rsp+10h] [rbp-D0h] BYREF
  int v24; // [rsp+18h] [rbp-C8h]
  _QWORD v25[2]; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v26[2]; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v27; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v28[2]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v29; // [rsp+70h] [rbp-70h] BYREF
  int v30; // [rsp+78h] [rbp-68h]
  _QWORD v31[2]; // [rsp+80h] [rbp-60h] BYREF
  __m128i v32; // [rsp+90h] [rbp-50h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 104);
  v5 = 1;
  if ( v4 )
  {
    _BitScanReverse64(&v6, v4);
    v7 = v6 ^ 0x3F;
    v4 = 0x8000000000000000LL >> v7;
    if ( 0x8000000000000000LL >> v7 > 9 )
    {
      if ( v4 <= 0x63 )
      {
        v5 = 2;
      }
      else if ( v4 <= 0x3E7 )
      {
        v5 = 3;
      }
      else if ( v4 <= 0x270F )
      {
        v5 = 4;
      }
      else
      {
        v8 = 0x8000000000000000LL >> v7;
        v9 = 1;
        do
        {
          v10 = v8;
          v11 = v9;
          v9 += 4;
          v8 /= 0x2710u;
          if ( v10 <= 0x1869F )
          {
            v5 = v9;
            goto LABEL_12;
          }
          if ( v10 <= 0xF423F )
          {
            v5 = v11 + 5;
            goto LABEL_12;
          }
          if ( v10 <= (unsigned __int64)&loc_98967F )
          {
            v5 = v11 + 6;
            goto LABEL_12;
          }
        }
        while ( v10 > 0x5F5E0FF );
        v5 = v11 + 7;
      }
    }
  }
LABEL_12:
  v29 = v31;
  sub_2240A50((__int64 *)&v29, v5, 0);
  sub_1249540(v29, v30, v4);
  v12 = *(_QWORD *)(a2 + 96);
  v13 = 1;
  if ( v12 )
  {
    _BitScanReverse64(&v14, v12);
    v15 = v14 ^ 0x3F;
    v12 = 0x8000000000000000LL >> v15;
    if ( 0x8000000000000000LL >> v15 > 9 )
    {
      if ( v12 <= 0x63 )
      {
        v13 = 2;
      }
      else if ( v12 <= 0x3E7 )
      {
        v13 = 3;
      }
      else if ( v12 <= 0x270F )
      {
        v13 = 4;
      }
      else
      {
        v16 = 0x8000000000000000LL >> v15;
        v17 = 1;
        do
        {
          v18 = v16;
          v19 = v17;
          v17 += 4;
          v16 /= 0x2710u;
          if ( v18 <= 0x1869F )
          {
            v13 = v17;
            goto LABEL_23;
          }
          if ( v18 <= 0xF423F )
          {
            v13 = v19 + 5;
            goto LABEL_23;
          }
          if ( v18 <= (unsigned __int64)&loc_98967F )
          {
            v13 = v19 + 6;
            goto LABEL_23;
          }
        }
        while ( v18 > 0x5F5E0FF );
        v13 = v19 + 7;
      }
    }
  }
LABEL_23:
  v23 = v25;
  sub_2240A50((__int64 *)&v23, v13, 0);
  sub_1249540(v23, v24, v12);
  v20 = (__m128i *)sub_2241130((unsigned __int64 *)&v23, 0, 0, "align<", 6u);
  v26[0] = (unsigned __int64)&v27;
  if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
  {
    v27 = _mm_loadu_si128(v20 + 1);
  }
  else
  {
    v26[0] = v20->m128i_i64[0];
    v27.m128i_i64[0] = v20[1].m128i_i64[0];
  }
  v21 = v20->m128i_u64[1];
  v20[1].m128i_i8[0] = 0;
  v26[1] = v21;
  v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
  v20->m128i_i64[1] = 0;
  sub_94F930(v28, (__int64)v26, "-");
  sub_8FD5D0(&v32, (__int64)v28, &v29);
  sub_94F930(a1, (__int64)&v32, ">");
  if ( (__int64 *)v32.m128i_i64[0] != &v33 )
    j_j___libc_free_0(v32.m128i_u64[0]);
  sub_2240A30((unsigned __int64 *)v28);
  sub_2240A30(v26);
  if ( v23 != (_BYTE *)v25 )
    j_j___libc_free_0((unsigned __int64)v23);
  if ( v29 != (_BYTE *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  return a1;
}
