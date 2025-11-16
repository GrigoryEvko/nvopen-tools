// Function: sub_2FE0E50
// Address: 0x2fe0e50
//
__m128i *__fastcall sub_2FE0E50(__m128i *a1, __int64 *a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // r10
  __int64 v10; // rax
  __int64 (__fastcall *v11)(__int64); // rcx
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 (*v16)(); // rcx
  __int64 v17; // rsi
  int v18; // eax
  __int64 *v19; // rbx
  __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rax
  _QWORD *v23; // r10
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  _DWORD *v28; // rsi
  __int64 v29; // rcx
  __int64 *v30; // rdi
  __int64 v31; // rcx
  __m128i *v32; // rdi
  __int64 *v33; // rsi
  __int64 v34; // rax
  const __m128i *v35; // rdx
  __int32 v37; // ebx
  __int64 v38; // rax
  __m128i v39; // xmm0
  __int64 v40; // rax
  __m128i v41; // xmm2
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-120h]
  _QWORD *v44; // [rsp+0h] [rbp-120h]
  __int64 v45; // [rsp+0h] [rbp-120h]
  __int64 v46; // [rsp+0h] [rbp-120h]
  __int64 v47; // [rsp+8h] [rbp-118h]
  _QWORD *v48; // [rsp+8h] [rbp-118h]
  char v49; // [rsp+1Fh] [rbp-101h] BYREF
  __int64 v50; // [rsp+20h] [rbp-100h] BYREF
  _DWORD *v51; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD v52[2]; // [rsp+30h] [rbp-F0h] BYREF
  char v53; // [rsp+40h] [rbp-E0h]
  __int32 v54; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+58h] [rbp-C8h]
  char v56; // [rsp+60h] [rbp-C0h]
  __int64 v57; // [rsp+70h] [rbp-B0h] BYREF
  bool v58; // [rsp+78h] [rbp-A8h]
  __int64 v59; // [rsp+98h] [rbp-88h]
  __m128i v60; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v61; // [rsp+B0h] [rbp-70h] BYREF
  __m128i v62; // [rsp+C0h] [rbp-60h] BYREF

  v7 = sub_2E88D60(a3);
  v47 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 16) + 200LL))(*(_QWORD *)(v7 + 16));
  v8 = (__int64 *)sub_B2BE50(*(_QWORD *)v7);
  v9 = sub_B0D000(v8, 0, 0, 0, 1);
  if ( *(_WORD *)(a3 + 68) == 20 )
  {
    v34 = *(_QWORD *)(a3 + 32);
    v35 = (const __m128i *)(v34 + 40);
    goto LABEL_30;
  }
  v10 = *a2;
  v11 = *(__int64 (__fastcall **)(__int64))(*a2 + 520);
  if ( v11 != sub_2DCA430 )
  {
    v45 = v9;
    ((void (__fastcall *)(_QWORD *, __int64 *, __int64))v11)(v52, a2, a3);
    v34 = v52[0];
    v35 = (const __m128i *)v52[1];
    v9 = v45;
    if ( !v53 )
    {
      v10 = *a2;
      goto LABEL_3;
    }
LABEL_30:
    if ( *(_DWORD *)(v34 + 8) == a4 )
    {
      v60 = _mm_loadu_si128(v35);
      v39 = _mm_loadu_si128(v35 + 1);
      *a1 = v60;
      v61 = v39;
      v40 = v35[2].m128i_i64[0];
      v62.m128i_i64[1] = v9;
      v62.m128i_i64[0] = v40;
      v41 = _mm_loadu_si128(&v62);
      a1[3].m128i_i8[0] = 1;
      a1[1] = v39;
      a1[2] = v41;
      return a1;
    }
    goto LABEL_31;
  }
LABEL_3:
  v12 = *(__int64 (__fastcall **)(__int64))(v10 + 544);
  if ( v12 != sub_2FDC580 )
  {
    v44 = (_QWORD *)v9;
    ((void (__fastcall *)(__int32 *, __int64 *, __int64, _QWORD))v12)(&v54, a2, a3, a4);
    v9 = (__int64)v44;
    if ( v56 )
    {
      v37 = v54;
      v50 = v55;
      v38 = sub_B0DAC0(v44, 0, v55);
      a1->m128i_i64[0] = 0;
      a1->m128i_i32[2] = v37;
      a1[1].m128i_i64[0] = 0;
      a1[1].m128i_i64[1] = 0;
      a1[2].m128i_i64[0] = 0;
      a1[2].m128i_i64[1] = v38;
      a1[3].m128i_i8[0] = 1;
      return a1;
    }
  }
  v13 = *(_QWORD *)(a3 + 48);
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_31;
  if ( (v13 & 7) != 0 )
  {
    if ( (v13 & 7) != 3 || *(_DWORD *)v14 != 1 )
      goto LABEL_31;
    v15 = 0;
    v16 = *(__int64 (**)())(**(_QWORD **)(v7 + 16) + 128LL);
    if ( v16 == sub_2DAC790 )
      goto LABEL_8;
  }
  else
  {
    *(_QWORD *)(a3 + 48) = v14;
    v15 = 0;
    v16 = *(__int64 (**)())(**(_QWORD **)(v7 + 16) + 128LL);
    if ( v16 == sub_2DAC790 )
      goto LABEL_7;
  }
  v46 = v9;
  v42 = v16();
  v9 = v46;
  v15 = (__int64 *)v42;
LABEL_7:
  v13 = *(_QWORD *)(a3 + 48);
  v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_44;
LABEL_8:
  v17 = *(_QWORD *)(v7 + 48);
  v18 = v13 & 7;
  if ( !v18 )
  {
    *(_QWORD *)(a3 + 48) = v14;
    v19 = (__int64 *)v14;
    goto LABEL_11;
  }
  if ( v18 != 3 )
LABEL_44:
    BUG();
  v19 = *(__int64 **)(v14 + 16);
LABEL_11:
  v20 = *v19;
  v43 = v9;
  if ( !*v19
    || (v20 & 4) == 0
    || (v21 = v20 & 0xFFFFFFFFFFFFFFF8LL) == 0
    || (*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v21 + 40LL))(v21, v17)
    || !(unsigned __int8)sub_2FE0930(v15, a3, &v51, (__int64)&v50, (__int64)&v49, v47)
    || v49
    || (unsigned int)sub_2E88FE0(a3) != 1 )
  {
LABEL_31:
    a1[3].m128i_i8[0] = 0;
    return a1;
  }
  v60.m128i_i64[1] = 0x800000000LL;
  v60.m128i_i64[0] = (__int64)&v61;
  sub_AF6280((__int64)&v60, v50);
  sub_A188E0((__int64)&v60, 148);
  v22 = sub_2EF2C60((__int64)v19);
  v23 = (_QWORD *)v43;
  if ( v22 == 0xBFFFFFFFFFFFFFFELL || v22 == -1 )
  {
    v26 = -1;
  }
  else
  {
    v24 = sub_2EF2C60((__int64)v19);
    v57 = v24 & 0x3FFFFFFFFFFFFFFFLL;
    v58 = (v24 & 0x4000000000000000LL) != 0;
    v25 = sub_CA1930(&v57);
    v23 = (_QWORD *)v43;
    v26 = v25;
  }
  v48 = v23;
  sub_A188E0((__int64)&v60, v26);
  v27 = sub_B0D8A0(v48, (__int64)&v60, 0, 0);
  v28 = v51;
  v29 = 10;
  v30 = &v57;
  while ( v29 )
  {
    *(_DWORD *)v30 = *v28++;
    v30 = (__int64 *)((char *)v30 + 4);
    --v29;
  }
  v31 = 12;
  v32 = a1;
  a1[3].m128i_i8[0] = 1;
  v59 = v27;
  v33 = &v57;
  while ( v31 )
  {
    v32->m128i_i32[0] = *(_DWORD *)v33;
    v33 = (__int64 *)((char *)v33 + 4);
    v32 = (__m128i *)((char *)v32 + 4);
    --v31;
  }
  if ( (__m128i *)v60.m128i_i64[0] != &v61 )
    _libc_free(v60.m128i_u64[0]);
  return a1;
}
