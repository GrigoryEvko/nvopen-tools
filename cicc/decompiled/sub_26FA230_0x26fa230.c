// Function: sub_26FA230
// Address: 0x26fa230
//
__int64 __fastcall sub_26FA230(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // r14
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r11d
  __int64 *v13; // r12
  unsigned int i; // r8d
  __int64 *v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 v18; // rax
  int v20; // ecx
  int v21; // ecx
  __m128i v22; // xmm0
  const __m128i *v23; // rbx
  unsigned __int64 v24; // rcx
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // r13
  __m128i *v28; // r13
  __int8 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int32 v40; // esi
  __int64 v41; // rax
  _QWORD *v42; // rdi
  int v43; // esi
  int v44; // esi
  __int64 v45; // rcx
  int v46; // r8d
  __int64 *v47; // rdi
  unsigned int j; // eax
  __int64 v49; // rdx
  unsigned int v50; // eax
  unsigned __int64 v51; // rdx
  const __m128i *v52; // rax
  __m128i *v53; // rdx
  __int32 v54; // r8d
  __int8 *v55; // rsi
  __int8 v56; // cl
  __int64 v57; // rcx
  __int64 v58; // rcx
  __int64 v59; // rax
  const __m128i *v60; // r14
  _QWORD *v61; // rdi
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  int v66; // eax
  int v67; // edx
  int v68; // edx
  __int64 v69; // rsi
  int v70; // r8d
  unsigned int k; // eax
  __int64 v72; // rcx
  unsigned int v73; // eax
  unsigned __int64 v74; // [rsp+0h] [rbp-170h]
  char v75; // [rsp+Fh] [rbp-161h]
  const __m128i *v76; // [rsp+18h] [rbp-158h]
  int v77; // [rsp+18h] [rbp-158h]
  unsigned __int64 v78[12]; // [rsp+28h] [rbp-148h] BYREF
  _QWORD v79[5]; // [rsp+88h] [rbp-E8h] BYREF
  __m128i v80; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 v81; // [rsp+C0h] [rbp-B0h]
  __int64 v82; // [rsp+C8h] [rbp-A8h]
  __int64 v83; // [rsp+D0h] [rbp-A0h]
  __int64 v84; // [rsp+D8h] [rbp-98h]
  unsigned __int64 v85; // [rsp+E0h] [rbp-90h]
  __int64 v86; // [rsp+E8h] [rbp-88h]
  __int64 v87; // [rsp+F0h] [rbp-80h]
  unsigned __int64 v88; // [rsp+F8h] [rbp-78h]
  __int64 v89; // [rsp+100h] [rbp-70h]
  __int64 v90; // [rsp+108h] [rbp-68h]
  __int64 v91; // [rsp+118h] [rbp-58h] BYREF
  _QWORD *v92; // [rsp+120h] [rbp-50h]
  __int64 *v93; // [rsp+128h] [rbp-48h]
  __int64 *v94; // [rsp+130h] [rbp-40h]
  __int64 v95; // [rsp+138h] [rbp-38h]

  v8 = a2->m128i_i64[0];
  v9 = a2->m128i_i64[1];
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_34;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v13 = 0;
  for ( i = (v10 - 1)
          & (((unsigned int)v8 >> 9)
           ^ ((unsigned int)v8 >> 4)
           ^ ((0xBF58476D1CE4E5B9LL * v9) >> 31)
           ^ (484763065 * v9)); ; i = (v10 - 1) & v17 )
  {
    v15 = (__int64 *)(v11 + 24LL * i);
    v16 = *v15;
    if ( *v15 == v8 && v15[1] == v9 )
    {
      v18 = *((unsigned int *)v15 + 4);
      return *(_QWORD *)(a1 + 32) + 144 * v18 + 16;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -2 && !v13 )
      v13 = (__int64 *)(v11 + 24LL * i);
LABEL_6:
    v17 = v12 + i;
    ++v12;
  }
  if ( v15[1] != -1 )
    goto LABEL_6;
  v20 = *(_DWORD *)(a1 + 16);
  if ( !v13 )
    v13 = (__int64 *)(v11 + 24LL * i);
  ++*(_QWORD *)a1;
  v21 = v20 + 1;
  if ( 4 * v21 >= 3 * v10 )
  {
LABEL_34:
    sub_26F8AB0(a1, 2 * v10);
    v43 = *(_DWORD *)(a1 + 24);
    if ( v43 )
    {
      v44 = v43 - 1;
      v46 = 1;
      v47 = 0;
      for ( j = v44
              & (((0xBF58476D1CE4E5B9LL * v9) >> 31)
               ^ (484763065 * v9)
               ^ ((unsigned int)v8 >> 9)
               ^ ((unsigned int)v8 >> 4)); ; j = v44 & v50 )
      {
        v45 = *(_QWORD *)(a1 + 8);
        v13 = (__int64 *)(v45 + 24LL * j);
        v49 = *v13;
        if ( *v13 == v8 && v13[1] == v9 )
          break;
        if ( v49 == -4096 )
        {
          if ( v13[1] == -1 )
          {
LABEL_79:
            if ( v47 )
              v13 = v47;
            v21 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_15;
          }
        }
        else if ( v49 == -8192 && v13[1] == -2 && !v47 )
        {
          v47 = (__int64 *)(v45 + 24LL * j);
        }
        v50 = v46 + j;
        ++v46;
      }
      goto LABEL_71;
    }
LABEL_92:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v21 <= v10 >> 3 )
  {
    sub_26F8AB0(a1, v10);
    v67 = *(_DWORD *)(a1 + 24);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 8);
      v47 = 0;
      v70 = 1;
      for ( k = v68
              & (((unsigned int)v8 >> 9)
               ^ ((unsigned int)v8 >> 4)
               ^ ((0xBF58476D1CE4E5B9LL * v9) >> 31)
               ^ (484763065 * v9)); ; k = v68 & v73 )
      {
        v13 = (__int64 *)(v69 + 24LL * k);
        v72 = *v13;
        if ( *v13 == v8 && v13[1] == v9 )
          break;
        if ( v72 == -4096 )
        {
          if ( v13[1] == -1 )
            goto LABEL_79;
        }
        else if ( v72 == -8192 && v13[1] == -2 && !v47 )
        {
          v47 = (__int64 *)(v69 + 24LL * k);
        }
        v73 = v70 + k;
        ++v70;
      }
LABEL_71:
      v21 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_15;
    }
    goto LABEL_92;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v21;
  if ( *v13 != -4096 || v13[1] != -1 )
    --*(_DWORD *)(a1 + 20);
  *v13 = v8;
  v13[1] = v9;
  *((_DWORD *)v13 + 4) = 0;
  v22 = _mm_loadu_si128(a2);
  v23 = &v80;
  v79[2] = v79;
  v79[3] = v79;
  memset(&v78[1], 0, 24);
  v78[4] = 1;
  memset(&v78[5], 0, 48);
  v79[0] = 0;
  v79[1] = 0;
  v79[4] = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 1;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v80 = v22;
  v93 = &v91;
  v24 = *(unsigned int *)(a1 + 44);
  v94 = &v91;
  v25 = *(unsigned int *)(a1 + 40);
  v95 = 0;
  v26 = v25;
  if ( v25 + 1 > v24 )
  {
    v51 = *(_QWORD *)(a1 + 32);
    if ( v51 > (unsigned __int64)&v80 || (unsigned __int64)&v80 >= v51 + 144 * v25 )
    {
      v74 = -1;
      v75 = 0;
    }
    else
    {
      v75 = 1;
      v74 = 0x8E38E38E38E38E39LL * ((__int64)((__int64)v80.m128i_i64 - v51) >> 4);
    }
    v27 = sub_C8D7D0(a1 + 32, a1 + 48, v25 + 1, 0x90u, v78, a6);
    v52 = *(const __m128i **)(a1 + 32);
    v76 = &v52[9 * *(unsigned int *)(a1 + 40)];
    if ( v52 != v76 )
    {
      v53 = (__m128i *)v27;
      do
      {
        if ( v53 )
        {
          v55 = &v53[6].m128i_i8[8];
          *v53 = _mm_loadu_si128(v52);
          v53[1].m128i_i64[0] = v52[1].m128i_i64[0];
          v53[1].m128i_i64[1] = v52[1].m128i_i64[1];
          v53[2].m128i_i64[0] = v52[2].m128i_i64[0];
          v56 = v52[2].m128i_i8[8];
          v52[2].m128i_i64[0] = 0;
          v52[1].m128i_i64[1] = 0;
          v52[1].m128i_i64[0] = 0;
          v53[2].m128i_i8[8] = v56;
          v53[2].m128i_i8[9] = v52[2].m128i_i8[9];
          v53[3].m128i_i64[0] = v52[3].m128i_i64[0];
          v53[3].m128i_i64[1] = v52[3].m128i_i64[1];
          v53[4].m128i_i64[0] = v52[4].m128i_i64[0];
          v57 = v52[4].m128i_i64[1];
          v52[4].m128i_i64[0] = 0;
          v52[3].m128i_i64[1] = 0;
          v52[3].m128i_i64[0] = 0;
          v53[4].m128i_i64[1] = v57;
          v53[5].m128i_i64[0] = v52[5].m128i_i64[0];
          v53[5].m128i_i64[1] = v52[5].m128i_i64[1];
          v52[5].m128i_i64[1] = 0;
          v52[5].m128i_i64[0] = 0;
          v52[4].m128i_i64[1] = 0;
          v58 = v52[7].m128i_i64[0];
          if ( v58 )
          {
            v54 = v52[6].m128i_i32[2];
            v53[7].m128i_i64[0] = v58;
            v53[6].m128i_i32[2] = v54;
            v53[7].m128i_i64[1] = v52[7].m128i_i64[1];
            v53[8].m128i_i64[0] = v52[8].m128i_i64[0];
            *(_QWORD *)(v58 + 8) = v55;
            v53[8].m128i_i64[1] = v52[8].m128i_i64[1];
            v52[7].m128i_i64[0] = 0;
            v52[7].m128i_i64[1] = (__int64)&v52[6].m128i_i64[1];
            v52[8].m128i_i64[0] = (__int64)&v52[6].m128i_i64[1];
            v52[8].m128i_i64[1] = 0;
          }
          else
          {
            v53[6].m128i_i32[2] = 0;
            v53[7].m128i_i64[0] = 0;
            v53[7].m128i_i64[1] = (__int64)v55;
            v53[8].m128i_i64[0] = (__int64)v55;
            v53[8].m128i_i64[1] = 0;
          }
        }
        v52 += 9;
        v53 += 9;
      }
      while ( v76 != v52 );
      v76 = *(const __m128i **)(a1 + 32);
      v59 = 9LL * *(unsigned int *)(a1 + 40);
      v60 = &v76[v59];
      if ( v76 != &v76[v59] )
      {
        do
        {
          v61 = (_QWORD *)v60[-2].m128i_i64[0];
          v60 -= 9;
          sub_26F6B10(v61);
          v62 = v60[4].m128i_u64[1];
          if ( v62 )
            j_j___libc_free_0(v62);
          v63 = v60[3].m128i_u64[0];
          if ( v63 )
            j_j___libc_free_0(v63);
          v64 = v60[1].m128i_u64[0];
          if ( v64 )
            j_j___libc_free_0(v64);
        }
        while ( v76 != v60 );
        v76 = *(const __m128i **)(a1 + 32);
      }
    }
    v65 = (unsigned __int64)v76;
    v66 = v78[0];
    if ( v76 != (const __m128i *)(a1 + 48) )
    {
      v77 = v78[0];
      _libc_free(v65);
      v66 = v77;
    }
    *(_DWORD *)(a1 + 44) = v66;
    v25 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v27;
    v26 = v25;
    if ( v75 )
      v23 = (const __m128i *)(v27 + 144 * v74);
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 32);
  }
  v28 = (__m128i *)(144 * v25 + v27);
  if ( v28 )
  {
    v29 = &v28[6].m128i_i8[8];
    *v28 = _mm_loadu_si128(v23);
    v30 = v23[1].m128i_i64[0];
    v23[1].m128i_i64[0] = 0;
    v28[1].m128i_i64[0] = v30;
    v31 = v23[1].m128i_i64[1];
    v23[1].m128i_i64[1] = 0;
    v28[1].m128i_i64[1] = v31;
    v32 = v23[2].m128i_i64[0];
    v23[2].m128i_i64[0] = 0;
    v28[2].m128i_i64[0] = v32;
    v28[2].m128i_i16[4] = v23[2].m128i_i16[4];
    v33 = v23[3].m128i_i64[0];
    v23[3].m128i_i64[0] = 0;
    v28[3].m128i_i64[0] = v33;
    v34 = v23[3].m128i_i64[1];
    v23[3].m128i_i64[1] = 0;
    v28[3].m128i_i64[1] = v34;
    v35 = v23[4].m128i_i64[0];
    v23[4].m128i_i64[0] = 0;
    v28[4].m128i_i64[0] = v35;
    v36 = v23[4].m128i_i64[1];
    v23[4].m128i_i64[1] = 0;
    v28[4].m128i_i64[1] = v36;
    v37 = v23[5].m128i_i64[0];
    v23[5].m128i_i64[0] = 0;
    v28[5].m128i_i64[0] = v37;
    v38 = v23[5].m128i_i64[1];
    v23[5].m128i_i64[1] = 0;
    v28[5].m128i_i64[1] = v38;
    v39 = v23[7].m128i_i64[0];
    if ( v39 )
    {
      v40 = v23[6].m128i_i32[2];
      v28[7].m128i_i64[0] = v39;
      v28[6].m128i_i32[2] = v40;
      v28[7].m128i_i64[1] = v23[7].m128i_i64[1];
      v28[8].m128i_i64[0] = v23[8].m128i_i64[0];
      *(_QWORD *)(v39 + 8) = v29;
      v41 = v23[8].m128i_i64[1];
      v23[7].m128i_i64[0] = 0;
      v28[8].m128i_i64[1] = v41;
      v23[7].m128i_i64[1] = (__int64)&v23[6].m128i_i64[1];
      v23[8].m128i_i64[0] = (__int64)&v23[6].m128i_i64[1];
      v23[8].m128i_i64[1] = 0;
    }
    else
    {
      v28[6].m128i_i32[2] = 0;
      v28[7].m128i_i64[0] = 0;
      v28[7].m128i_i64[1] = (__int64)v29;
      v28[8].m128i_i64[0] = (__int64)v29;
      v28[8].m128i_i64[1] = 0;
    }
    v26 = *(_DWORD *)(a1 + 40);
  }
  v42 = v92;
  *(_DWORD *)(a1 + 40) = v26 + 1;
  sub_26F6B10(v42);
  if ( v88 )
    j_j___libc_free_0(v88);
  if ( v85 )
    j_j___libc_free_0(v85);
  if ( v81 )
    j_j___libc_free_0(v81);
  sub_26F6B10(0);
  v18 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *((_DWORD *)v13 + 4) = v18;
  return *(_QWORD *)(a1 + 32) + 144 * v18 + 16;
}
