// Function: sub_E9E970
// Address: 0xe9e970
//
__int64 __fastcall sub_E9E970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rcx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned int *v20; // rbx
  int v21; // r11d
  __int64 v22; // rcx
  __int64 v23; // r15
  unsigned int v24; // edx
  __int64 v25; // rax
  unsigned int *v26; // r10
  __int64 v27; // rdi
  __int64 v28; // rdi
  __m128i *v29; // rsi
  int v31; // eax
  int v32; // edx
  __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  int v36; // eax
  __int64 v37; // rdx
  unsigned int **v38; // rcx
  unsigned int **v39; // rdx
  unsigned int *v40; // rax
  unsigned int *v41; // rax
  unsigned int *v42; // rax
  __int64 v43; // r8
  __int64 v44; // rsi
  __int64 v45; // rsi
  int v46; // eax
  int v47; // esi
  __int64 v48; // rdi
  unsigned int v49; // eax
  __int64 v50; // r8
  unsigned __int64 v51; // r14
  __int64 v52; // r8
  int v53; // eax
  __int64 v54; // rdi
  int v55; // r8d
  __int64 v56; // rsi
  unsigned int v57; // r14d
  __int64 v58; // rax
  unsigned int v59; // [rsp+Ch] [rbp-84h] BYREF
  unsigned int *v60; // [rsp+10h] [rbp-80h] BYREF
  __int64 v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __int64 v63; // [rsp+28h] [rbp-68h]
  __m128i v64; // [rsp+30h] [rbp-60h] BYREF
  __m128i v65; // [rsp+40h] [rbp-50h] BYREF
  __m128i v66; // [rsp+50h] [rbp-40h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v7 = sub_E6C430(v6, a2, a3, a4, a5);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_QWORD *)(v6 + 1776);
  v65.m128i_i64[0] = 0;
  v11 = v7;
  v12 = *(_QWORD *)(v6 + 1784);
  v66.m128i_i64[0] = 0;
  v13 = *(_DWORD *)(v9 + 1912);
  v64.m128i_i64[0] = v10;
  v64.m128i_i64[1] = v12;
  v14 = *(_QWORD *)(v9 + 1744);
  v15 = v9 + 1736;
  v65.m128i_i64[1] = v11;
  v66.m128i_i8[8] = 0;
  v59 = v13;
  if ( !v14 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v14 + 16);
      v17 = *(_QWORD *)(v14 + 24);
      if ( v13 <= *(_DWORD *)(v14 + 32) )
        break;
      v14 = *(_QWORD *)(v14 + 24);
      if ( !v17 )
        goto LABEL_6;
    }
    v15 = v14;
    v14 = *(_QWORD *)(v14 + 16);
  }
  while ( v16 );
LABEL_6:
  if ( v9 + 1736 == v15 || v13 < *(_DWORD *)(v15 + 32) )
  {
LABEL_16:
    v60 = &v59;
    v15 = sub_E9E2A0((_QWORD *)(v9 + 1728), v15, &v60);
  }
  v18 = *(_DWORD *)(v15 + 584);
  v19 = v15 + 560;
  v20 = *(unsigned int **)(*(_QWORD *)(a1 + 288) + 8LL);
  if ( !v18 )
  {
    ++*(_QWORD *)(v15 + 560);
    goto LABEL_38;
  }
  v21 = 1;
  v22 = *(_QWORD *)(v15 + 568);
  v23 = 0;
  v24 = (v18 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
  v25 = v22 + 16LL * v24;
  v26 = *(unsigned int **)v25;
  if ( v20 == *(unsigned int **)v25 )
  {
LABEL_10:
    v27 = *(unsigned int *)(v25 + 8);
    goto LABEL_11;
  }
  while ( v26 != (unsigned int *)-4096LL )
  {
    if ( !v23 && v26 == (unsigned int *)-8192LL )
      v23 = v25;
    v8 = (unsigned int)(v21 + 1);
    v24 = (v18 - 1) & (v21 + v24);
    v25 = v22 + 16LL * v24;
    v26 = *(unsigned int **)v25;
    if ( v20 == *(unsigned int **)v25 )
      goto LABEL_10;
    ++v21;
  }
  if ( !v23 )
    v23 = v25;
  v31 = *(_DWORD *)(v15 + 576);
  ++*(_QWORD *)(v15 + 560);
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v18 )
  {
LABEL_38:
    sub_E7B220(v19, 2 * v18);
    v46 = *(_DWORD *)(v15 + 584);
    if ( v46 )
    {
      v47 = v46 - 1;
      v48 = *(_QWORD *)(v15 + 568);
      v49 = (v46 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v32 = *(_DWORD *)(v15 + 576) + 1;
      v23 = v48 + 16LL * v49;
      v33 = *(_QWORD *)v23;
      if ( v20 != *(unsigned int **)v23 )
      {
        v8 = 1;
        v50 = 0;
        while ( v33 != -4096 )
        {
          if ( !v50 && v33 == -8192 )
            v50 = v23;
          v49 = v47 & (v8 + v49);
          v23 = v48 + 16LL * v49;
          v33 = *(_QWORD *)v23;
          if ( v20 == *(unsigned int **)v23 )
            goto LABEL_27;
          v8 = (unsigned int)(v8 + 1);
        }
        if ( v50 )
          v23 = v50;
      }
      goto LABEL_27;
    }
    goto LABEL_65;
  }
  v33 = v18 >> 3;
  if ( v18 - *(_DWORD *)(v15 + 580) - v32 <= (unsigned int)v33 )
  {
    sub_E7B220(v19, v18);
    v53 = *(_DWORD *)(v15 + 584);
    if ( v53 )
    {
      v33 = (unsigned int)(v53 - 1);
      v54 = *(_QWORD *)(v15 + 568);
      v55 = 1;
      v56 = 0;
      v57 = v33 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v32 = *(_DWORD *)(v15 + 576) + 1;
      v23 = v54 + 16LL * v57;
      v58 = *(_QWORD *)v23;
      if ( v20 != *(unsigned int **)v23 )
      {
        while ( v58 != -4096 )
        {
          if ( v58 == -8192 && !v56 )
            v56 = v23;
          v8 = (unsigned int)(v55 + 1);
          v57 = v33 & (v55 + v57);
          v23 = v54 + 16LL * v57;
          v58 = *(_QWORD *)v23;
          if ( v20 == *(unsigned int **)v23 )
            goto LABEL_27;
          ++v55;
        }
        if ( v56 )
          v23 = v56;
      }
      goto LABEL_27;
    }
LABEL_65:
    ++*(_DWORD *)(v15 + 576);
    BUG();
  }
LABEL_27:
  *(_DWORD *)(v15 + 576) = v32;
  if ( *(_QWORD *)v23 != -4096 )
    --*(_DWORD *)(v15 + 580);
  *(_QWORD *)v23 = v20;
  *(_DWORD *)(v23 + 8) = 0;
  v60 = v20;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v27 = *(unsigned int *)(v15 + 600);
  v34 = *(unsigned int *)(v15 + 604);
  v35 = v27 + 1;
  v36 = v27;
  if ( v27 + 1 > v34 )
  {
    v51 = *(_QWORD *)(v15 + 592);
    v52 = v15 + 592;
    if ( v51 > (unsigned __int64)&v60 || (unsigned __int64)&v60 >= v51 + 32 * v27 )
    {
      sub_E79C70(v15 + 592, v35, v34, v33, v52, v8);
      v27 = *(unsigned int *)(v15 + 600);
      v37 = *(_QWORD *)(v15 + 592);
      v38 = &v60;
      v36 = *(_DWORD *)(v15 + 600);
    }
    else
    {
      sub_E79C70(v15 + 592, v35, v34, v33, v52, v8);
      v37 = *(_QWORD *)(v15 + 592);
      v27 = *(unsigned int *)(v15 + 600);
      v38 = (unsigned int **)((char *)&v60 + v37 - v51);
      v36 = *(_DWORD *)(v15 + 600);
    }
  }
  else
  {
    v37 = *(_QWORD *)(v15 + 592);
    v38 = &v60;
  }
  v39 = (unsigned int **)(32 * v27 + v37);
  if ( v39 )
  {
    *v39 = *v38;
    v40 = v38[1];
    v38[1] = 0;
    v39[1] = v40;
    v41 = v38[2];
    v38[2] = 0;
    v39[2] = v41;
    v42 = v38[3];
    v38[3] = 0;
    v39[3] = v42;
    v27 = *(unsigned int *)(v15 + 600);
    v43 = v61;
    v44 = v63;
    v36 = *(_DWORD *)(v15 + 600);
    *(_DWORD *)(v15 + 600) = v27 + 1;
    v45 = v44 - v43;
    if ( v43 )
    {
      j_j___libc_free_0(v43, v45);
      v27 = (unsigned int)(*(_DWORD *)(v15 + 600) - 1);
      v36 = *(_DWORD *)(v15 + 600) - 1;
    }
  }
  else
  {
    *(_DWORD *)(v15 + 600) = v36 + 1;
  }
  *(_DWORD *)(v23 + 8) = v36;
LABEL_11:
  v28 = *(_QWORD *)(v15 + 592) + 32 * v27;
  v29 = *(__m128i **)(v28 + 16);
  if ( v29 == *(__m128i **)(v28 + 24) )
  {
    sub_E782B0((const __m128i **)(v28 + 8), v29, &v64);
  }
  else
  {
    if ( v29 )
    {
      *v29 = _mm_loadu_si128(&v64);
      v29[1] = _mm_loadu_si128(&v65);
      v29[2] = _mm_loadu_si128(&v66);
      v29 = *(__m128i **)(v28 + 16);
    }
    *(_QWORD *)(v28 + 16) = v29 + 3;
  }
  return v11;
}
