// Function: sub_1DA2EF0
// Address: 0x1da2ef0
//
void __fastcall sub_1DA2EF0(__int64 a1, __int64 a2, __int64 **a3, __int64 a4, __int64 a5, int a6, int a7)
{
  __int64 v10; // r14
  __int64 v11; // r12
  int v12; // eax
  _WORD *v13; // rdx
  _BOOL4 v14; // ecx
  __int64 v15; // rdx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r11
  __int64 v19; // r12
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  const __m128i *v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __m128i *v28; // rax
  __m128i *v29; // r12
  __m128i *v30; // rdx
  unsigned __int64 v31; // rdx
  bool v32; // cl
  unsigned __int64 v33; // rdx
  bool v34; // cf
  unsigned int v35; // esi
  __m128i *v36; // r13
  const __m128i *v37; // rsi
  __int64 v38; // rax
  _QWORD *v39; // rax
  int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rax
  _BYTE *v43; // rsi
  int v45; // [rsp+0h] [rbp-E0h]
  __int64 v46; // [rsp+0h] [rbp-E0h]
  __int64 v47; // [rsp+8h] [rbp-D8h]
  __int64 v48; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v49; // [rsp+10h] [rbp-D0h]
  __m128i *v50; // [rsp+28h] [rbp-B8h] BYREF
  __m128i v51; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+40h] [rbp-A0h]
  const __m128i *v53; // [rsp+48h] [rbp-98h] BYREF
  __int64 v54; // [rsp+50h] [rbp-90h]
  __int64 v55; // [rsp+58h] [rbp-88h] BYREF
  _BYTE *v56; // [rsp+60h] [rbp-80h]
  _BYTE *v57; // [rsp+68h] [rbp-78h]
  __int64 v58; // [rsp+70h] [rbp-70h]
  int v59; // [rsp+78h] [rbp-68h]
  _BYTE v60[32]; // [rsp+80h] [rbp-60h] BYREF
  int v61; // [rsp+A0h] [rbp-40h]
  unsigned __int64 v62; // [rsp+A8h] [rbp-38h]

  v49 = (unsigned __int64)(unsigned int)(a6 - 1) << 7;
  v10 = *(_QWORD *)(*(_QWORD *)(a5 + 48) + v49 + 16);
  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 56LL);
  if ( a7 )
  {
    v47 = sub_1E16510(v10);
    v12 = sub_1E16500(v10);
    v13 = *(_WORD **)(v10 + 16);
    v14 = 0;
    if ( *v13 == 12 )
    {
      v43 = *(_BYTE **)(v10 + 32);
      if ( !*v43 )
        v14 = v43[40] == 1;
    }
    sub_1E1C0A0(v11, v10 + 64, (_DWORD)v13, v14, a7, v12, v47);
    v18 = a2;
    v19 = v15;
    if ( **(_WORD **)(v10 + 16) == 12 )
    {
      v42 = *(_QWORD *)(v10 + 32);
      if ( !*(_BYTE *)v42 && *(_BYTE *)(v42 + 40) == 1 )
        *(_QWORD *)(*(_QWORD *)(v15 + 32) + 64LL) = *(_QWORD *)(v42 + 64);
    }
    v20 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v20 < *(_DWORD *)(a4 + 12) )
      goto LABEL_5;
LABEL_42:
    v48 = v18;
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v16, v17);
    v20 = *(unsigned int *)(a4 + 8);
    v18 = v48;
    goto LABEL_5;
  }
  v38 = ***(_QWORD ***)(a2 + 56);
  if ( (v38 & 4) == 0 )
    BUG();
  v45 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, __m128i *))(**(_QWORD **)(a1 + 248) + 176LL))(
          *(_QWORD *)(a1 + 248),
          v11,
          *(unsigned int *)((v38 & 0xFFFFFFFFFFFFFFF8LL) + 16),
          &v51);
  v39 = (_QWORD *)sub_1E16510(v10);
  v46 = sub_15C48E0(v39, 0, v45, 0, 0);
  v40 = sub_1E16500(v10);
  sub_1E1C0A0(v11, v10 + 64, *(_QWORD *)(v10 + 16), 1, v51.m128i_i32[0], v40, v46);
  v18 = a2;
  v19 = v41;
  v20 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v20 >= *(_DWORD *)(a4 + 12) )
    goto LABEL_42;
LABEL_5:
  v21 = (_QWORD *)(*(_QWORD *)a4 + 16 * v20);
  v22 = 0;
  *v21 = v18;
  v21[1] = v19;
  ++*(_DWORD *)(a4 + 8);
  sub_1DA10C0(a3, *(__int64 **)(*(_QWORD *)(a5 + 48) + v49), *(__int64 **)(*(_QWORD *)(a5 + 48) + v49 + 8));
  v23 = sub_15C70A0(v19 + 64);
  if ( *(_DWORD *)(v23 + 8) == 2 )
    v22 = *(_QWORD *)(v23 - 8);
  v24 = sub_1E16500(v19);
  v51.m128i_i64[1] = v22;
  v51.m128i_i64[0] = v24;
  v52 = v19;
  v25 = *(const __m128i **)(v19 + 64);
  v50 = (__m128i *)v25;
  if ( v25 )
  {
    sub_1623A60((__int64)&v50, (__int64)v25, 2);
    v53 = v50;
    if ( v50 )
      sub_1623210((__int64)&v50, (unsigned __int8 *)v50, (__int64)&v53);
  }
  else
  {
    v53 = 0;
  }
  v54 = a1 + 280;
  v55 = 0;
  v56 = v60;
  v57 = v60;
  v58 = 4;
  v59 = 0;
  v61 = 0;
  v26 = *(_QWORD *)(v19 + 32);
  if ( !*(_BYTE *)v26 )
  {
    v27 = *(int *)(v26 + 8);
    if ( (_DWORD)v27 )
    {
      v61 = 1;
      v62 = v27;
    }
  }
  v28 = *(__m128i **)(a5 + 16);
  if ( !v28 )
  {
    v29 = (__m128i *)(a5 + 8);
    goto LABEL_32;
  }
  v29 = (__m128i *)(a5 + 8);
  while ( 1 )
  {
    v31 = v28[2].m128i_u64[0];
    v32 = v31 < v51.m128i_i64[0];
    if ( v31 == v51.m128i_i64[0] )
    {
      v33 = v28[2].m128i_u64[1];
      v32 = v33 < v51.m128i_i64[1];
      if ( v33 == v51.m128i_i64[1] )
        v32 = v28[9].m128i_i64[1] < v62;
    }
    v30 = (__m128i *)v28[1].m128i_i64[1];
    if ( !v32 )
    {
      v30 = (__m128i *)v28[1].m128i_i64[0];
      v29 = v28;
    }
    if ( !v30 )
      break;
    v28 = v30;
  }
  if ( v29 == (__m128i *)(a5 + 8) )
  {
LABEL_32:
    v50 = &v51;
    v29 = sub_1DA0CF0((_QWORD *)a5, v29, (const __m128i **)&v50);
    v35 = v29[10].m128i_u32[0];
    if ( v35 )
      goto LABEL_26;
    goto LABEL_33;
  }
  v34 = v51.m128i_i64[0] < (unsigned __int64)v29[2].m128i_i64[0];
  if ( v51.m128i_i64[0] != v29[2].m128i_i64[0]
    || (v34 = v51.m128i_i64[1] < (unsigned __int64)v29[2].m128i_i64[1], v51.m128i_i64[1] != v29[2].m128i_i64[1]) )
  {
    if ( !v34 )
      goto LABEL_25;
    goto LABEL_32;
  }
  if ( v62 < v29[9].m128i_i64[1] )
    goto LABEL_32;
LABEL_25:
  v35 = v29[10].m128i_u32[0];
  if ( v35 )
    goto LABEL_26;
LABEL_33:
  v29[10].m128i_i32[0] = ((__int64)(*(_QWORD *)(a5 + 56) - *(_QWORD *)(a5 + 48)) >> 7) + 1;
  v36 = *(__m128i **)(a5 + 56);
  if ( v36 == *(__m128i **)(a5 + 64) )
  {
    sub_1DA0880((const __m128i **)(a5 + 48), *(const __m128i **)(a5 + 56), &v51);
  }
  else
  {
    if ( v36 )
    {
      *v36 = _mm_load_si128(&v51);
      v36[1].m128i_i64[0] = v52;
      v37 = v53;
      v36[1].m128i_i64[1] = (__int64)v53;
      if ( v37 )
        sub_1623A60((__int64)&v36[1].m128i_i64[1], (__int64)v37, 2);
      v36[2].m128i_i64[0] = v54;
      sub_16CCCB0(&v36[2].m128i_i64[1], (__int64)v36[5].m128i_i64, (__int64)&v55);
      v36[7].m128i_i32[0] = v61;
      v36[7].m128i_i64[1] = v62;
      v36 = *(__m128i **)(a5 + 56);
    }
    *(_QWORD *)(a5 + 56) = v36 + 8;
  }
  v35 = v29[10].m128i_u32[0];
LABEL_26:
  sub_1DA2AD0((__m128i *)a3, v35, v51.m128i_i64[0], v51.m128i_i64[1]);
  if ( v57 != v56 )
    _libc_free((unsigned __int64)v57);
  if ( v53 )
    sub_161E7C0((__int64)&v53, (__int64)v53);
}
