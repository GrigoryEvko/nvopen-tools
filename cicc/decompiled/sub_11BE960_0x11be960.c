// Function: sub_11BE960
// Address: 0x11be960
//
__int64 __fastcall sub_11BE960(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned __int8 *a8)
{
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int8 *v16; // r13
  unsigned int v17; // r14d
  __m128i v18; // xmm1
  unsigned __int64 v19; // r15
  __int64 v21; // rax
  __m128i v22; // xmm0
  __int64 v23; // r13
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int8 v30; // al
  __m128i v31; // [rsp+0h] [rbp-280h] BYREF
  unsigned __int8 *v32; // [rsp+10h] [rbp-270h]
  char v33; // [rsp+23h] [rbp-25Dh] BYREF
  __int32 v34; // [rsp+24h] [rbp-25Ch] BYREF
  __int64 v35; // [rsp+28h] [rbp-258h] BYREF
  __m128i v36; // [rsp+30h] [rbp-250h] BYREF
  unsigned __int8 *v37; // [rsp+40h] [rbp-240h]
  char v38[32]; // [rsp+50h] [rbp-230h] BYREF
  __int64 v39[4]; // [rsp+70h] [rbp-210h] BYREF
  _QWORD v40[2]; // [rsp+90h] [rbp-1F0h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-1E0h]
  __int64 v42; // [rsp+A8h] [rbp-1D8h] BYREF
  unsigned int v43; // [rsp+B0h] [rbp-1D0h]
  _QWORD v44[2]; // [rsp+168h] [rbp-118h] BYREF
  _BYTE v45[192]; // [rsp+178h] [rbp-108h] BYREF
  __int64 v46; // [rsp+238h] [rbp-48h]
  __int64 v47; // [rsp+240h] [rbp-40h]
  __int64 v48; // [rsp+248h] [rbp-38h]

  v40[1] = 0;
  v41 = 1;
  v40[0] = sub_B43CA0(a2);
  v10 = &v42;
  do
  {
    *v10 = -4096;
    v10 += 3;
    *((_DWORD *)v10 - 4) = 100;
  }
  while ( v10 != v44 );
  v46 = a2;
  v48 = a4;
  v44[0] = v45;
  v44[1] = 0x800000000LL;
  v47 = a3;
  v11 = sub_B43CC0(a2);
  sub_11BE290(&v31, v11, v12, v13, v14, v15, a7, a8);
  v16 = v32;
  v17 = v31.m128i_i32[0];
  v18 = _mm_loadu_si128(&v31);
  a8 = v32;
  a7 = (__int128)v18;
  if ( !v31.m128i_i32[0] )
    goto LABEL_10;
  if ( !v32 )
    goto LABEL_17;
  v19 = *((_QWORD *)&a7 + 1);
  if ( *(_BYTE *)(*((_QWORD *)v32 + 1) + 8LL) != 14 )
    goto LABEL_6;
  v11 = 6;
  v30 = *sub_98ACB0(v32, 6u);
  if ( v30 > 0x1Cu )
  {
    if ( v30 != 60 )
      goto LABEL_6;
LABEL_10:
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_11;
  }
  if ( v30 <= 3u )
    goto LABEL_10;
LABEL_6:
  if ( *v16 == 22 )
  {
    v11 = v17;
    if ( (unsigned __int8)sub_B2D670((__int64)v16, v17) )
    {
      if ( v17 - 86 > 0xA )
        goto LABEL_10;
      v11 = v17;
      v39[0] = sub_B2D8E0((__int64)v16, v17);
      if ( v19 <= sub_A71B80(v39) )
        goto LABEL_10;
    }
    goto LABEL_17;
  }
  if ( *v16 <= 0x1Cu || (v11 = 0, !sub_F509B0(v16, 0)) )
  {
LABEL_17:
    v21 = v46;
    goto LABEL_18;
  }
  if ( !*((_QWORD *)v16 + 2) )
    goto LABEL_10;
  v29 = sub_BD3700((__int64)v16);
  v21 = v46;
  if ( v29 )
  {
    if ( *(_QWORD *)(v29 + 24) == v46 )
      goto LABEL_10;
  }
LABEL_18:
  v22 = _mm_loadu_si128((const __m128i *)&a7);
  v11 = (__int64)a8;
  v36 = v22;
  v37 = a8;
  if ( v21 && a8 )
  {
    v39[0] = (__int64)v40;
    v39[1] = (__int64)&v36;
    v39[2] = (__int64)&v33;
    v39[3] = (__int64)&v35;
    v34 = v36.m128i_i32[0];
    v33 = 0;
    v35 = 0;
    sub_CF9460(
      (__int64)v38,
      (__int64)a8,
      &v34,
      1,
      v47,
      (__int64)&v34,
      (unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64))sub_11BE1D0,
      (__int64)v39);
    v23 = v35;
    if ( v35 )
    {
      v24 = v36.m128i_i64[1];
      v25 = sub_BCB2E0(*(_QWORD **)v40[0]);
      v11 = v24;
      v26 = sub_ACD640(v25, v24, 0);
      if ( *(_QWORD *)v23 )
      {
        v27 = *(_QWORD *)(v23 + 8);
        **(_QWORD **)(v23 + 16) = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = *(_QWORD *)(v23 + 16);
      }
      *(_QWORD *)v23 = v26;
      if ( v26 )
      {
        v28 = *(_QWORD *)(v26 + 16);
        *(_QWORD *)(v23 + 8) = v28;
        if ( v28 )
        {
          v11 = v23 + 8;
          *(_QWORD *)(v28 + 16) = v23 + 8;
        }
        *(_QWORD *)(v23 + 16) = v26 + 16;
        *(_QWORD *)(v26 + 16) = v23;
      }
    }
    if ( v33 )
      goto LABEL_10;
    v22 = _mm_loadu_si128((const __m128i *)&a7);
    v11 = (__int64)a8;
  }
  *(_QWORD *)(a1 + 16) = v11;
  *(__m128i *)a1 = v22;
LABEL_11:
  if ( (_BYTE *)v44[0] != v45 )
    _libc_free(v44[0], v11);
  if ( (v41 & 1) == 0 )
    sub_C7D6A0(v42, 24LL * v43, 8);
  return a1;
}
