// Function: sub_3996FA0
// Address: 0x3996fa0
//
void __fastcall sub_3996FA0(__m128i *a1, __int64 a2, int *a3, size_t a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // esi
  __int64 v9; // rdi
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  int v13; // r10d
  int v14; // edx
  __int32 v15; // eax
  __m128i *v16; // rax
  __int64 v17; // r14
  __int64 v18; // rax
  _QWORD *v19; // r12
  unsigned __int32 v20; // edx
  _QWORD *v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rdi
  unsigned __int32 v25; // r8d
  __m128i *v26; // r14
  unsigned __int64 v27; // rbx
  __int64 v28; // rax
  __m128i *v29; // rax
  __m128i *v30; // rdx
  __int64 v31; // rsi
  __int64 *v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // r14
  __int64 v39; // r12
  __int64 v40; // r14
  _QWORD *v41; // r13
  bool v42; // zf
  __int32 v43; // edx
  __int64 v44; // rdi
  __int32 v45; // esi
  __int64 v46; // r8
  unsigned int v47; // ecx
  __int64 *v48; // rdx
  __int64 v49; // r9
  __m128i *v50; // rbx
  __m128i *v51; // r12
  _QWORD *v52; // r13
  __int32 v53; // ecx
  __m128i *v54; // r12
  __int64 v55; // rsi
  __m128i *v56; // r15
  _QWORD *v57; // r12
  int v58; // edx
  int v59; // r10d
  __int32 v60; // eax
  __int32 v61; // esi
  __int64 v62; // rcx
  unsigned int v63; // eax
  __int64 v64; // rdi
  __int64 *v65; // r10
  int v66; // r9d
  __int64 *v67; // r8
  __int32 v68; // eax
  __int32 v69; // ecx
  __int64 v70; // rsi
  unsigned int v71; // ebx
  __int64 v72; // rax
  __int64 *v73; // r9
  int v74; // r8d
  __int64 *v75; // rdi
  __int32 v76; // [rsp+Ch] [rbp-84h]
  unsigned __int32 v77; // [rsp+Ch] [rbp-84h]
  __m128i *v78; // [rsp+10h] [rbp-80h]
  _QWORD *v79; // [rsp+10h] [rbp-80h]
  _QWORD *v80; // [rsp+18h] [rbp-78h]
  __int64 *v81; // [rsp+18h] [rbp-78h]
  __int64 v82; // [rsp+18h] [rbp-78h]
  __int64 v83; // [rsp+18h] [rbp-78h]
  unsigned __int64 v86; // [rsp+28h] [rbp-68h]
  __m128i *v89; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v90; // [rsp+48h] [rbp-48h]
  _BYTE v91[64]; // [rsp+50h] [rbp-40h] BYREF

  if ( a1[279].m128i_i32[2] && a1[346].m128i_i8[8] )
    return;
  v8 = a1[278].m128i_u32[2];
  if ( v8 )
  {
    v9 = a1[277].m128i_i64[1];
    v10 = (v8 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
    v11 = (_QWORD *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == a6 )
    {
LABEL_5:
      sub_39A3940(a2, a5, v11[1]);
      return;
    }
    v80 = 0;
    v13 = 1;
    while ( v12 != -8 )
    {
      if ( !v80 )
      {
        if ( v12 != -16 )
          v11 = 0;
        v80 = v11;
      }
      v10 = (v8 - 1) & (v13 + v10);
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v11 == a6 )
        goto LABEL_5;
      ++v13;
    }
    if ( v80 )
      v11 = v80;
    ++a1[277].m128i_i64[0];
    v81 = v11;
    v14 = a1[278].m128i_i32[0] + 1;
    if ( 4 * v14 < 3 * v8 )
    {
      if ( v8 - a1[278].m128i_i32[1] - v14 > v8 >> 3 )
        goto LABEL_13;
      sub_3996DE0((__int64)a1[277].m128i_i64, v8);
      v68 = a1[278].m128i_i32[2];
      if ( v68 )
      {
        v69 = v68 - 1;
        v70 = a1[277].m128i_i64[1];
        v71 = (v68 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        v14 = a1[278].m128i_i32[0] + 1;
        v81 = (__int64 *)(v70 + 16LL * v71);
        v72 = *v81;
        if ( *v81 != a6 )
        {
          v73 = (__int64 *)(v70 + 16LL * v71);
          v74 = 1;
          v75 = 0;
          while ( v72 != -8 )
          {
            if ( !v75 && v72 == -16 )
              v75 = v73;
            v71 = v69 & (v74 + v71);
            v73 = (__int64 *)(v70 + 16LL * v71);
            v72 = *v73;
            if ( *v73 == a6 )
            {
              v81 = (__int64 *)(v70 + 16LL * v71);
              goto LABEL_13;
            }
            ++v74;
          }
          if ( !v75 )
            v75 = v73;
          v81 = v75;
        }
        goto LABEL_13;
      }
LABEL_113:
      ++a1[278].m128i_i32[0];
      BUG();
    }
  }
  else
  {
    ++a1[277].m128i_i64[0];
  }
  sub_3996DE0((__int64)a1[277].m128i_i64, 2 * v8);
  v60 = a1[278].m128i_i32[2];
  if ( !v60 )
    goto LABEL_113;
  v61 = v60 - 1;
  v62 = a1[277].m128i_i64[1];
  v63 = (v60 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
  v81 = (__int64 *)(v62 + 16LL * v63);
  v64 = *v81;
  v14 = a1[278].m128i_i32[0] + 1;
  if ( *v81 != a6 )
  {
    v65 = (__int64 *)(v62 + 16LL * (v61 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
    v66 = 1;
    v67 = 0;
    while ( v64 != -8 )
    {
      if ( !v67 && v64 == -16 )
        v67 = v65;
      v63 = v61 & (v66 + v63);
      v65 = (__int64 *)(v62 + 16LL * v63);
      v64 = *v65;
      if ( *v65 == a6 )
      {
        v81 = (__int64 *)(v62 + 16LL * v63);
        goto LABEL_13;
      }
      ++v66;
    }
    if ( !v67 )
      v67 = v65;
    v81 = v67;
  }
LABEL_13:
  a1[278].m128i_i32[0] = v14;
  if ( *v81 != -8 )
    --a1[278].m128i_i32[1];
  *v81 = a6;
  v81[1] = 0;
  v15 = a1[279].m128i_i32[2];
  a1[346].m128i_i8[8] = 0;
  v76 = v15;
  v16 = sub_398BB50(a1, a2);
  v17 = a1->m128i_i64[1];
  v78 = v16;
  v18 = sub_22077B0(0x280u);
  v19 = (_QWORD *)v18;
  if ( v18 )
    sub_39A2080(v18, a2, v17, a1, &a1[252].m128i_u64[1], v78);
  v20 = a1[279].m128i_u32[2];
  if ( v20 >= a1[279].m128i_i32[3] )
  {
    sub_398EE70((__int64)a1[279].m128i_i64, 0);
    v20 = a1[279].m128i_u32[2];
  }
  v79 = v19;
  v21 = (_QWORD *)(a1[279].m128i_i64[0] + 16LL * v20);
  if ( v21 )
  {
    v79 = 0;
    *v21 = v19;
    v21[1] = a6;
    v20 = a1[279].m128i_u32[2];
  }
  a1[279].m128i_i32[2] = v20 + 1;
  v22 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 80) + 24LL);
  LODWORD(v89) = 65541;
  sub_39A3560(v19, v19 + 2, 19, &v89, v22);
  v23 = sub_398C040(a3, a4);
  v19[75] = v23;
  v81[1] = v23;
  v86 = v23;
  v24 = a1->m128i_i64[1];
  if ( a1[282].m128i_i8[1] )
  {
    v19[7] = *(_QWORD *)(sub_396DD80(v24) + 232);
  }
  else
  {
    v28 = sub_396DD80(v24);
    v19[7] = sub_38D3660(v28 + 8, v86);
    sub_39C9BF0(a2, v19 + 1);
  }
  if ( a1[282].m128i_i8[2] && !a1[282].m128i_i8[1] )
    sub_39A3E90(v19);
  v19[76] = sub_39A92A0(v19, a6);
  if ( v76 )
  {
LABEL_32:
    sub_39A3940(a2, a5, v86);
    v27 = (unsigned __int64)v79;
    if ( v79 )
    {
      *v79 = &unk_4A3FCC0;
LABEL_34:
      sub_39A20E0(v27);
      j_j___libc_free_0(v27);
      return;
    }
    return;
  }
  v25 = a1[279].m128i_u32[2];
  v89 = (__m128i *)v91;
  v90 = 0x100000000LL;
  if ( v25 )
  {
    v29 = (__m128i *)a1[279].m128i_i64[0];
    v30 = a1 + 280;
    v31 = v25;
    if ( v29 == &a1[280] )
    {
      v32 = (__int64 *)v91;
      v33 = 1;
      if ( v25 != 1 )
      {
        v77 = v25;
        v82 = v25;
        sub_398EE70((__int64)&v89, v25);
        v32 = (__int64 *)v89;
        v31 = v82;
        v30 = (__m128i *)a1[279].m128i_i64[0];
        v33 = a1[279].m128i_u32[2];
        v25 = v77;
      }
      v34 = 2 * v33;
      if ( 16 * v33 )
      {
        v35 = (__int64 *)v30;
        v36 = &v32[v34];
        do
        {
          if ( v32 )
          {
            *v32 = *v35;
            v37 = v35[1];
            *v35 = 0;
            v32[1] = v37;
          }
          v32 += 2;
          v35 += 2;
        }
        while ( v36 != v32 );
        v38 = a1[279].m128i_u32[2];
        v39 = a1[279].m128i_i64[0];
        LODWORD(v90) = v25;
        v40 = v39 + 16 * v38;
        if ( v40 != v39 )
        {
          v83 = a2;
          do
          {
            v41 = *(_QWORD **)(v40 - 16);
            v40 -= 16;
            if ( v41 )
            {
              *v41 = &unk_4A3FCC0;
              sub_39A20E0(v41);
              j_j___libc_free_0((unsigned __int64)v41);
            }
          }
          while ( v39 != v40 );
          a2 = v83;
          v31 = (unsigned int)v90;
        }
      }
      else
      {
        LODWORD(v90) = v25;
      }
      v29 = v89;
      v26 = &v89[v31];
    }
    else
    {
      v53 = a1[279].m128i_i32[3];
      v89 = (__m128i *)a1[279].m128i_i64[0];
      v90 = __PAIR64__(v53, v25);
      v26 = &v29[v25];
      a1[279].m128i_i64[0] = (__int64)v30;
      a1[279].m128i_i32[3] = 0;
    }
    v42 = a1[346].m128i_i8[8] == 0;
    a1[279].m128i_i32[2] = 0;
    if ( v42 )
    {
      if ( v29 != v26 )
      {
        v54 = v29;
        do
        {
          v55 = v54->m128i_i64[0];
          ++v54;
          sub_39A02B0(&a1[252].m128i_u64[1], v55);
          sub_39A01D0(&a1[252].m128i_u64[1], v54[-1].m128i_i64[0], a1[282].m128i_u8[1]);
        }
        while ( v26 != v54 );
        v56 = v89;
        v26 = &v89[(unsigned int)v90];
        if ( v89 != v26 )
        {
          do
          {
            v57 = (_QWORD *)v26[-1].m128i_i64[0];
            --v26;
            if ( v57 )
            {
              *v57 = &unk_4A3FCC0;
              sub_39A20E0(v57);
              j_j___libc_free_0((unsigned __int64)v57);
            }
          }
          while ( v56 != v26 );
          v26 = v89;
        }
      }
      goto LABEL_30;
    }
    for ( ; v29 != v26; ++v29 )
    {
      v43 = a1[278].m128i_i32[2];
      if ( v43 )
      {
        v44 = v29->m128i_i64[1];
        v45 = v43 - 1;
        v46 = a1[277].m128i_i64[1];
        v47 = (v43 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v48 = (__int64 *)(v46 + 16LL * v47);
        v49 = *v48;
        if ( v44 == *v48 )
        {
LABEL_61:
          *v48 = -16;
          --a1[278].m128i_i32[0];
          ++a1[278].m128i_i32[1];
        }
        else
        {
          v58 = 1;
          while ( v49 != -8 )
          {
            v59 = v58 + 1;
            v47 = v45 & (v58 + v47);
            v48 = (__int64 *)(v46 + 16LL * v47);
            v49 = *v48;
            if ( v44 == *v48 )
              goto LABEL_61;
            v58 = v59;
          }
        }
      }
    }
  }
  else if ( !a1[346].m128i_i8[8] )
  {
    v26 = (__m128i *)v91;
LABEL_30:
    if ( v26 != (__m128i *)v91 )
      _libc_free((unsigned __int64)v26);
    goto LABEL_32;
  }
  sub_39A8AE0(a2, a5, a6);
  v50 = v89;
  v51 = &v89[(unsigned int)v90];
  if ( v89 != v51 )
  {
    do
    {
      v52 = (_QWORD *)v51[-1].m128i_i64[0];
      --v51;
      if ( v52 )
      {
        *v52 = &unk_4A3FCC0;
        sub_39A20E0(v52);
        j_j___libc_free_0((unsigned __int64)v52);
      }
    }
    while ( v50 != v51 );
    v51 = v89;
  }
  if ( v51 != (__m128i *)v91 )
    _libc_free((unsigned __int64)v51);
  v27 = (unsigned __int64)v79;
  if ( v79 )
  {
    *v79 = &unk_4A3FCC0;
    goto LABEL_34;
  }
}
