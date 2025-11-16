// Function: sub_3978030
// Address: 0x3978030
//
void __fastcall sub_3978030(_QWORD **a1, __int64 a2, _BYTE *a3, char a4)
{
  __int64 v4; // r15
  __int64 v8; // rsi
  __int64 v9; // rcx
  _BYTE *v10; // rdx
  __int64 v11; // rax
  _BYTE *v12; // r10
  _BYTE *i; // rbx
  unsigned __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // rdx
  int v17; // r8d
  __int64 v18; // r9
  __int64 v19; // r11
  __int64 v20; // rax
  __m128i *v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // ecx
  __int64 v24; // rdx
  unsigned int v25; // ecx
  unsigned __int64 *v26; // rcx
  __int64 v27; // rax
  unsigned int v28; // eax
  __m128i *v29; // rbx
  __int64 v30; // rax
  char *v31; // r10
  __int64 v32; // rcx
  __m128i *v33; // rax
  __m128i *v34; // r15
  __m128i *v35; // rdi
  __m128i *v36; // rax
  __m128i v37; // xmm0
  __int64 v38; // rsi
  const __m128i *v39; // rax
  __m128i *v40; // rbx
  __int64 (__fastcall *v41)(__int64); // rax
  __int64 v42; // rsi
  _QWORD *v43; // rdx
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 *v46; // rdx
  __int64 (*v47)(); // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 *v50; // r15
  __int64 v51; // rax
  __int64 v52; // [rsp-150h] [rbp-150h]
  __m128i *v53; // [rsp-150h] [rbp-150h]
  __int64 v54; // [rsp-148h] [rbp-148h]
  __int64 v55; // [rsp-148h] [rbp-148h]
  __int64 v56; // [rsp-140h] [rbp-140h]
  __int64 v57; // [rsp-140h] [rbp-140h]
  __int64 v58; // [rsp-140h] [rbp-140h]
  __int64 v59; // [rsp-140h] [rbp-140h]
  __int64 v60; // [rsp-140h] [rbp-140h]
  __int64 v61; // [rsp-138h] [rbp-138h]
  unsigned int v62; // [rsp-138h] [rbp-138h]
  __int64 v63; // [rsp-138h] [rbp-138h]
  unsigned int v64; // [rsp-138h] [rbp-138h]
  _BYTE *v66; // [rsp-130h] [rbp-130h]
  __int64 v67; // [rsp-130h] [rbp-130h]
  __m128i *v68; // [rsp-130h] [rbp-130h]
  __m128i v69; // [rsp-128h] [rbp-128h] BYREF
  __int64 v70; // [rsp-118h] [rbp-118h]
  __m128i *v71; // [rsp-108h] [rbp-108h] BYREF
  __int64 v72; // [rsp-100h] [rbp-100h]
  _BYTE v73[248]; // [rsp-F8h] [rbp-F8h] BYREF

  if ( a3[16] != 6 )
    return;
  v4 = *(_QWORD *)(*(_QWORD *)a3 + 24LL);
  if ( *(_BYTE *)(v4 + 8) != 13 )
    return;
  if ( (unsigned int)(*(_DWORD *)(v4 + 12) - 2) > 1 )
    return;
  if ( *(_BYTE *)(sub_1643D80(*(_QWORD *)(*(_QWORD *)a3 + 24LL), 0) + 8) != 11 )
    return;
  v8 = 1;
  if ( *(_BYTE *)(sub_1643D80(v4, 1u) + 8) != 15 )
    return;
  v10 = a3;
  if ( *(_DWORD *)(v4 + 12) == 3 )
  {
    v8 = 2;
    if ( *(_BYTE *)(sub_1643D80(v4, 2u) + 8) != 15 )
      return;
    v10 = a3;
  }
  v71 = (__m128i *)v73;
  v72 = 0x800000000LL;
  v11 = 24LL * (*((_DWORD *)v10 + 5) & 0xFFFFFFF);
  if ( (v10[23] & 0x40) != 0 )
  {
    v12 = (_BYTE *)*((_QWORD *)v10 - 1);
    v66 = &v12[v11];
  }
  else
  {
    v66 = v10;
    v12 = &v10[-v11];
  }
  for ( i = v12; v66 != i; i += 24 )
  {
    v15 = *(_QWORD *)i;
    if ( *(_BYTE *)(*(_QWORD *)i + 16LL) == 7 )
    {
      v61 = *(_QWORD *)i;
      v16 = 1LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF);
      if ( sub_1593BB0(*(_QWORD *)(v15 + 24 * v16), v8, v16, v9) )
        break;
      v18 = v61;
      v19 = *(_QWORD *)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v19 + 16) == 13 )
      {
        v69.m128i_i32[0] = 0;
        v20 = (unsigned int)v72;
        v69.m128i_i64[1] = 0;
        v70 = 0;
        if ( (unsigned int)v72 >= HIDWORD(v72) )
        {
          v8 = (__int64)v73;
          v60 = v19;
          sub_16CD150((__int64)&v71, v73, 0, 24, v17, v61);
          v20 = (unsigned int)v72;
          v19 = v60;
          v18 = v61;
        }
        v21 = (__m128i *)((char *)v71 + 24 * v20);
        v22 = v70;
        *v21 = _mm_loadu_si128(&v69);
        v21[1].m128i_i64[0] = v22;
        v23 = *(_DWORD *)(v19 + 32);
        LODWORD(v72) = v72 + 1;
        v62 = v23;
        v24 = (__int64)&v71[-1] + 24 * (unsigned int)v72 - 8;
        if ( v23 <= 0x40 )
        {
          v14 = *(_QWORD *)(v19 + 24);
          if ( v14 > 0xFFFF )
            LODWORD(v14) = 0xFFFF;
        }
        else
        {
          v54 = (__int64)&v71[-1] + 24 * (unsigned int)v72 - 8;
          v56 = v18;
          v52 = v19;
          LODWORD(v14) = sub_16A57B0(v19 + 24);
          v18 = v56;
          v24 = v54;
          v25 = v62 - v14;
          LODWORD(v14) = 0xFFFF;
          if ( v25 <= 0x40 )
          {
            LODWORD(v14) = 0xFFFF;
            v26 = *(unsigned __int64 **)(v52 + 24);
            if ( *v26 <= 0xFFFF )
              v14 = *v26;
          }
        }
        *(_DWORD *)v24 = v14;
        v9 = 1LL - (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
        *(_QWORD *)(v24 + 8) = *(_QWORD *)(v18 + 24 * v9);
        if ( *(_DWORD *)(v4 + 12) == 3 )
        {
          v57 = v24;
          v63 = v18;
          v8 = 2LL - (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
          if ( !sub_1593BB0(*(_QWORD *)(v18 + 24 * v8), v8, v24, 2) )
          {
            v27 = sub_1649C60(*(_QWORD *)(v63 + 24 * (2LL - (*(_DWORD *)(v63 + 20) & 0xFFFFFFF))));
            v9 = 0;
            if ( *(_BYTE *)(v27 + 16) >= 4u )
              v27 = 0;
            *(_QWORD *)(v57 + 16) = v27;
          }
        }
      }
    }
  }
  v28 = sub_15A94D0(a2, 0);
  v64 = -1;
  if ( v28 )
  {
    _BitScanReverse(&v28, v28);
    v64 = 31 - (v28 ^ 0x1F);
  }
  v29 = v71;
  v30 = 24LL * (unsigned int)v72;
  v31 = &v71->m128i_i8[v30];
  v32 = 0xAAAAAAAAAAAAAAABLL * (v30 >> 3);
  if ( v30 )
  {
    while ( 1 )
    {
      v53 = (__m128i *)v31;
      v55 = v32;
      v58 = 24 * v32;
      v67 = 24 * v32;
      v33 = (__m128i *)sub_2207800(24 * v32);
      v31 = (char *)v53;
      v34 = v33;
      if ( v33 )
        break;
      v32 = v55 >> 1;
      if ( !(v55 >> 1) )
        goto LABEL_65;
    }
    v35 = (__m128i *)((char *)v33 + v67);
    *v33 = _mm_loadu_si128(v29);
    v33[1].m128i_i64[0] = v29[1].m128i_i64[0];
    v36 = (__m128i *)((char *)v33 + 24);
    if ( v35 == (__m128i *)&v34[1].m128i_u64[1] )
    {
      v39 = v34;
    }
    else
    {
      do
      {
        v37 = _mm_loadu_si128((__m128i *)((char *)v36 - 24));
        v38 = v36[-1].m128i_i64[1];
        v36 = (__m128i *)((char *)v36 + 24);
        *(__m128i *)((char *)v36 - 24) = v37;
        v36[-1].m128i_i64[1] = v38;
      }
      while ( v35 != v36 );
      v39 = (__m128i *)((char *)v34 + v58 - 24);
    }
    *v29 = _mm_loadu_si128(v39);
    v29[1].m128i_i64[0] = v39[1].m128i_i64[0];
    sub_396CF40(v29, v53, v34, (const __m128i *)v55);
  }
  else
  {
LABEL_65:
    v34 = 0;
    sub_396C7E0(v29->m128i_i8, v31);
  }
  j_j___libc_free_0((unsigned __int64)v34);
  v40 = v71;
  v68 = (__m128i *)((char *)v71 + 24 * (unsigned int)v72);
  if ( v71 == v68 )
    goto LABEL_57;
  do
  {
    v48 = sub_396DD80((__int64)a1);
    v49 = v40[1].m128i_i64[0];
    v50 = (__int64 *)v48;
    if ( v49 )
    {
      if ( (*(_BYTE *)(v49 + 32) & 0xF) == 1 )
        goto LABEL_48;
      v59 = v40[1].m128i_i64[0];
      if ( sub_15E4F60(v59) )
        goto LABEL_48;
      v49 = sub_396EAF0((__int64)a1, v59);
    }
    v51 = *v50;
    if ( a4 )
    {
      v41 = *(__int64 (__fastcall **)(__int64))(v51 + 96);
      if ( v41 == sub_21C97E0 )
      {
        v42 = v50[98];
        goto LABEL_43;
      }
    }
    else
    {
      v41 = *(__int64 (__fastcall **)(__int64))(v51 + 104);
      if ( v41 == sub_21C97F0 )
      {
        v42 = v50[99];
        goto LABEL_43;
      }
    }
    v42 = ((__int64 (__fastcall *)(__int64 *, _QWORD, __int64))v41)(v50, v40->m128i_u32[0], v49);
LABEL_43:
    (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1[32] + 160LL))(a1[32], v42, 0);
    v43 = a1[32];
    v44 = *((unsigned int *)v43 + 30);
    if ( (_DWORD)v44 )
    {
      v45 = (_QWORD *)(v43[14] + 32 * v44 - 32);
      if ( v45[2] != *v45 || v45[1] != v45[3] )
        sub_396F480((__int64)a1, v64, 0);
    }
    v46 = (__int64 *)v40->m128i_i64[1];
    v47 = (__int64 (*)())(*a1)[39];
    if ( v47 == sub_214ADC0 )
      sub_3976960((__int64)a1, a2, v46);
    else
      ((void (__fastcall *)(_QWORD **, __int64, __int64 *))v47)(a1, a2, v46);
LABEL_48:
    v40 = (__m128i *)((char *)v40 + 24);
  }
  while ( v68 != v40 );
  v68 = v71;
LABEL_57:
  if ( v68 != (__m128i *)v73 )
    _libc_free((unsigned __int64)v68);
}
