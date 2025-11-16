// Function: sub_11C1FA0
// Address: 0x11c1fa0
//
char __fastcall sub_11C1FA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned __int8 *a8)
{
  unsigned __int64 v8; // rax
  unsigned __int8 *v9; // r12
  int v10; // r13d
  __m128i v11; // xmm0
  unsigned __int64 v12; // r14
  __int64 v13; // r8
  __int64 v14; // r12
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rdx
  char v23; // [rsp-8h] [rbp-C0h]
  __m128i v24; // [rsp+8h] [rbp-B0h] BYREF
  unsigned __int8 *v25; // [rsp+18h] [rbp-A0h]
  char v26; // [rsp+2Bh] [rbp-8Dh] BYREF
  __int32 v27; // [rsp+2Ch] [rbp-8Ch] BYREF
  __int64 v28; // [rsp+30h] [rbp-88h] BYREF
  __m128i v29; // [rsp+38h] [rbp-80h] BYREF
  unsigned __int8 *v30; // [rsp+48h] [rbp-70h]
  _BYTE v31[32]; // [rsp+58h] [rbp-60h] BYREF
  __m128i v32; // [rsp+78h] [rbp-40h] BYREF
  char *v33; // [rsp+88h] [rbp-30h]
  __int64 *v34; // [rsp+90h] [rbp-28h]

  LOBYTE(v8) = (unsigned __int8)sub_11BE290(&v24, *(_QWORD *)a1 + 312LL, a3, a4, a5, a6, a7, a8);
  v9 = v25;
  v10 = v24.m128i_i32[0];
  v11 = _mm_loadu_si128(&v24);
  a8 = v25;
  a7 = (__int128)v11;
  if ( !v24.m128i_i32[0] )
    return v8;
  if ( v25 )
  {
    v12 = *((_QWORD *)&a7 + 1);
    if ( *(_BYTE *)(*((_QWORD *)v25 + 1) + 8LL) == 14 )
    {
      LOBYTE(v8) = *sub_98ACB0(v25, 6u);
      if ( (unsigned __int8)v8 > 0x1Cu )
      {
        if ( (_BYTE)v8 == 60 )
          return v8;
      }
      else if ( (unsigned __int8)v8 <= 3u )
      {
        return v8;
      }
    }
    if ( *v9 == 22 )
    {
      if ( (unsigned __int8)sub_B2D670((__int64)v9, v10) )
      {
        LOBYTE(v8) = v10 - 86;
        if ( (unsigned int)(v10 - 86) > 0xA )
          return v8;
        v32.m128i_i64[0] = sub_B2D8E0((__int64)v9, v10);
        v8 = sub_A71B80(v32.m128i_i64);
        if ( v12 <= v8 )
          return v8;
      }
      v8 = *(_QWORD *)(a1 + 424);
      goto LABEL_11;
    }
    if ( *v9 > 0x1Cu )
    {
      LOBYTE(v8) = sub_F509B0(v9, 0);
      if ( (_BYTE)v8 )
      {
        if ( !*((_QWORD *)v9 + 2) )
          return v8;
        v21 = sub_BD3700((__int64)v9);
        v8 = *(_QWORD *)(a1 + 424);
        if ( v21 )
        {
          if ( *(_QWORD *)(v21 + 24) == v8 )
            return v8;
        }
        goto LABEL_11;
      }
    }
  }
  v8 = *(_QWORD *)(a1 + 424);
LABEL_11:
  v29 = _mm_loadu_si128((const __m128i *)&a7);
  v30 = a8;
  if ( !v8 || !a8 )
    goto LABEL_22;
  v32.m128i_i64[1] = (__int64)&v29;
  v33 = &v26;
  v13 = *(_QWORD *)(a1 + 432);
  v34 = &v28;
  v27 = v29.m128i_i32[0];
  v26 = 0;
  v28 = 0;
  v32.m128i_i64[0] = a1;
  sub_CF9460(
    (__int64)v31,
    (__int64)a8,
    &v27,
    1,
    v13,
    (__int64)&v27,
    (unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64))sub_11BE1D0,
    (__int64)&v32);
  v14 = v28;
  LOBYTE(v8) = v23;
  if ( v28 )
  {
    v15 = v29.m128i_i64[1];
    v16 = sub_BCB2E0(**(_QWORD ***)a1);
    v8 = sub_ACD640(v16, v15, 0);
    if ( *(_QWORD *)v14 )
    {
      v17 = *(_QWORD *)(v14 + 8);
      **(_QWORD **)(v14 + 16) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v14 + 16);
    }
    *(_QWORD *)v14 = v8;
    if ( v8 )
    {
      v18 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v14 + 8) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = v14 + 8;
      *(_QWORD *)(v14 + 16) = v8 + 16;
      *(_QWORD *)(v8 + 16) = v14;
    }
  }
  if ( !v26 )
  {
LABEL_22:
    v32.m128i_i64[0] = (__int64)a8;
    v32.m128i_i32[2] = a7;
    v8 = sub_11C1A30((const __m128i *)(a1 + 8), &v32, (__int64 *)&a7 + 1);
    if ( !v19 )
    {
      v20 = *((_QWORD *)&a7 + 1);
      if ( *(_QWORD *)(v8 + 16) >= *((_QWORD *)&a7 + 1) )
        v20 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v8 + 16) = v20;
    }
  }
  return v8;
}
