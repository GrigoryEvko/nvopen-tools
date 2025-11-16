// Function: sub_1BD7CA0
// Address: 0x1bd7ca0
//
_QWORD **__fastcall sub_1BD7CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  __int64 *v9; // r12
  __int64 v10; // rax
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 i; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rdx
  void *v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // rax
  _QWORD **result; // rax
  __int64 v30; // r9
  int v31; // eax
  __int64 v32; // r13
  __int64 v33; // r12
  __int64 v34; // rax
  _DWORD *v35; // rax
  __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned int v39; // ecx
  __int64 *v40; // rdx
  __int64 v41; // r8
  unsigned int v42; // eax
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // r15
  __int64 v46; // rdx
  int v47; // r8d
  unsigned int v48; // edi
  __int64 v49; // rax
  __int64 v50; // rdi
  _QWORD **v51; // rax
  _QWORD *v52; // rax
  int v53; // esi
  __int64 v54; // rbx
  char v55; // cl
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rcx
  _QWORD *v59; // rsi
  __int64 v60; // rdx
  __int64 v61; // rcx
  _QWORD *v62; // rdx
  __int64 v63; // rax
  __m128i *v64; // rax
  __int64 v65; // rax
  int v66; // eax
  int v67; // eax
  int v68; // edx
  int v69; // r10d
  bool v70; // zf
  _QWORD *v71; // rax
  __int64 v72; // rdx
  unsigned int v73; // eax
  unsigned int v74; // ebx
  char v75; // al
  __int64 v76; // rdi
  __int64 v77; // rax
  _QWORD *v78; // rax
  __int64 v79; // rdx
  const void *v80; // [rsp+8h] [rbp-98h]
  _QWORD **v81; // [rsp+10h] [rbp-90h]
  __int64 v82; // [rsp+20h] [rbp-80h]
  __int64 v83; // [rsp+20h] [rbp-80h]
  __int64 v84; // [rsp+28h] [rbp-78h]
  __m128i v85; // [rsp+30h] [rbp-70h] BYREF
  int v86; // [rsp+4Ch] [rbp-54h] BYREF
  __m128i v87; // [rsp+50h] [rbp-50h] BYREF
  __int64 v88; // [rsp+60h] [rbp-40h]

  v9 = (__int64 *)a2;
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned __int64 **)(a1 + 8);
  v84 = a4;
  v85.m128i_i64[0] = a5;
  v85.m128i_i64[1] = a6;
  v82 = v10;
  if ( (unsigned __int64 *)v10 != v11 )
  {
    v12 = (unsigned __int64 *)v10;
    do
    {
      v13 = v12[19];
      if ( (unsigned __int64 *)v13 != v12 + 21 )
        _libc_free(v13);
      v14 = v12[12];
      if ( (unsigned __int64 *)v14 != v12 + 14 )
        _libc_free(v14);
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        _libc_free(*v12);
      v12 += 22;
    }
    while ( v11 != v12 );
    *(_QWORD *)(a1 + 8) = v82;
  }
  v15 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 24);
  v16 = v15 >> 1;
  if ( v16 )
  {
    if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    {
      a4 = 4 * v16;
      goto LABEL_14;
    }
LABEL_116:
    v19 = (_QWORD *)(a1 + 40);
    v20 = 8;
    goto LABEL_17;
  }
  i = *(unsigned int *)(a1 + 36);
  if ( !(_DWORD)i )
    goto LABEL_20;
  a4 = 0;
  if ( (*(_BYTE *)(a1 + 32) & 1) != 0 )
    goto LABEL_116;
LABEL_14:
  v18 = *(unsigned int *)(a1 + 48);
  if ( (unsigned int)v18 <= (unsigned int)a4 || (unsigned int)v18 <= 0x40 )
  {
    v19 = *(_QWORD **)(a1 + 40);
    v20 = 2 * v18;
LABEL_17:
    for ( i = (__int64)&v19[v20]; (_QWORD *)i != v19; v19 += 2 )
      *v19 = -8;
    *(_QWORD *)(a1 + 32) &= 1uLL;
    goto LABEL_20;
  }
  if ( !v16 || (v73 = v16 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    *(_BYTE *)(a1 + 32) |= 1u;
    goto LABEL_120;
  }
  _BitScanReverse(&v73, v73);
  a4 = 33 - (v73 ^ 0x1F);
  v74 = 1 << (33 - (v73 ^ 0x1F));
  if ( v74 - 5 <= 0x3A )
  {
    v74 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    v75 = *(_BYTE *)(a1 + 32);
    v76 = 1024;
    goto LABEL_132;
  }
  if ( (_DWORD)v18 != v74 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 40));
    v75 = *(_BYTE *)(a1 + 32) | 1;
    *(_BYTE *)(a1 + 32) = v75;
    if ( v74 <= 4 )
      goto LABEL_120;
    v76 = 16LL * v74;
LABEL_132:
    *(_BYTE *)(a1 + 32) = v75 & 0xFE;
    v77 = sub_22077B0(v76);
    *(_DWORD *)(a1 + 48) = v74;
    *(_QWORD *)(a1 + 40) = v77;
LABEL_120:
    v70 = (*(_QWORD *)(a1 + 32) & 1LL) == 0;
    *(_QWORD *)(a1 + 32) &= 1uLL;
    if ( v70 )
    {
      v71 = *(_QWORD **)(a1 + 40);
      v72 = 2LL * *(unsigned int *)(a1 + 48);
    }
    else
    {
      v71 = (_QWORD *)(a1 + 40);
      v72 = 8;
    }
    for ( i = (__int64)&v71[v72]; (_QWORD *)i != v71; v71 += 2 )
    {
      if ( v71 )
        *v71 = -8;
    }
    goto LABEL_20;
  }
  v70 = (*(_QWORD *)(a1 + 32) & 1LL) == 0;
  *(_QWORD *)(a1 + 32) &= 1uLL;
  if ( v70 )
  {
    v78 = *(_QWORD **)(a1 + 40);
    v79 = 2 * v18;
  }
  else
  {
    v78 = (_QWORD *)(a1 + 40);
    v79 = 8;
  }
  i = (__int64)&v78[v79];
  do
  {
    if ( v78 )
      *v78 = -8;
    v78 += 2;
  }
  while ( (_QWORD *)i != v78 );
LABEL_20:
  ++*(_QWORD *)(a1 + 104);
  v21 = *(void **)(a1 + 120);
  if ( v21 == *(void **)(a1 + 112) )
  {
LABEL_25:
    *(_QWORD *)(a1 + 132) = 0;
    goto LABEL_26;
  }
  v22 = 4 * (*(_DWORD *)(a1 + 132) - *(_DWORD *)(a1 + 136));
  v23 = *(unsigned int *)(a1 + 128);
  if ( v22 < 0x20 )
    v22 = 32;
  if ( (unsigned int)v23 <= v22 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(v21, -1, 8 * v23);
    goto LABEL_25;
  }
  sub_16CC920(a1 + 104);
LABEL_26:
  *(_DWORD *)(a1 + 392) = 0;
  sub_1BC21B0(a1 + 1264, a2, i, a4, a5, a6);
  v24 = *(_QWORD *)(a1 + 1224);
  v25 = *(_QWORD *)(a1 + 1232);
  for ( *(_DWORD *)(a1 + 1296) = 0; v25 != v24; *(_DWORD *)(v26 + 216) = 0 )
  {
    v26 = *(_QWORD *)(v24 + 8);
    v27 = *(_DWORD *)(v26 + 220) - *(_DWORD *)(v26 + 216);
    *(_DWORD *)(v26 + 112) = 0;
    *(_QWORD *)(v26 + 184) = 0;
    *(_QWORD *)(v26 + 192) = 0;
    if ( v27 < 16 )
      v27 = 16;
    v24 += 16;
    ++*(_DWORD *)(v26 + 224);
    *(_QWORD *)(v26 + 200) = 0;
    *(_QWORD *)(v26 + 208) = 0;
    *(_DWORD *)(v26 + 220) = v27;
  }
  sub_196A810(a1 + 1472);
  v28 = *(_QWORD *)(a1 + 1504);
  if ( v28 != *(_QWORD *)(a1 + 1512) )
    *(_QWORD *)(a1 + 1512) = v28;
  *(__m128i *)(a1 + 1248) = _mm_load_si128(&v85);
  if ( (int)a3 <= 1 )
  {
LABEL_39:
    sub_1BD5480(a1, v9, a3, 0, -1, a7);
    result = *(_QWORD ***)(a1 + 8);
    v81 = result;
    v85.m128i_i64[0] = *(_QWORD *)a1;
    if ( (_QWORD **)v85.m128i_i64[0] == result )
      return result;
    while ( 1 )
    {
      if ( !*(_BYTE *)(v85.m128i_i64[0] + 88) )
      {
        v31 = *(_DWORD *)(v85.m128i_i64[0] + 8);
        if ( v31 )
          break;
      }
LABEL_41:
      v85.m128i_i64[0] += 176;
      result = (_QWORD **)v85.m128i_i64[0];
      if ( v81 == (_QWORD **)v85.m128i_i64[0] )
        return result;
    }
    v32 = 0;
    v83 = (unsigned int)(v31 - 1);
    v80 = (const void *)(a1 + 400);
    while ( 1 )
    {
      v33 = *(_QWORD *)(*(_QWORD *)v85.m128i_i64[0] + 8 * v32);
      v86 = v32;
      v34 = *(unsigned int *)(v85.m128i_i64[0] + 104);
      if ( (_DWORD)v34 )
      {
        v35 = sub_1BB97F0(*(_DWORD **)(v85.m128i_i64[0] + 96), *(_QWORD *)(v85.m128i_i64[0] + 96) + 4 * v34, &v86);
        v86 = ((__int64)v35 - v36) >> 2;
      }
      v37 = *(unsigned int *)(v84 + 24);
      if ( (_DWORD)v37 )
      {
        v38 = *(_QWORD *)(v84 + 8);
        v39 = (v37 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v40 = (__int64 *)(v38 + 16LL * v39);
        v41 = *v40;
        if ( v33 == *v40 )
        {
LABEL_49:
          if ( v40 != (__int64 *)(v38 + 16 * v37)
            && *(_QWORD *)(v84 + 40) != *(_QWORD *)(v84 + 32) + 40LL * *((unsigned int *)v40 + 2) )
          {
            v42 = *(_DWORD *)(a1 + 392);
            if ( v42 >= *(_DWORD *)(a1 + 396) )
            {
              sub_16CD150(a1 + 384, v80, 0, 24, v41, v30);
              v42 = *(_DWORD *)(a1 + 392);
            }
            v43 = *(_QWORD *)(a1 + 384) + 24LL * v42;
            if ( v43 )
            {
              v44 = v86;
              *(_QWORD *)(v43 + 8) = 0;
              *(_QWORD *)v43 = v33;
              *(_DWORD *)(v43 + 16) = v44;
              v42 = *(_DWORD *)(a1 + 392);
            }
            *(_DWORD *)(a1 + 392) = v42 + 1;
          }
        }
        else
        {
          v68 = 1;
          while ( v41 != -8 )
          {
            LODWORD(v30) = v68 + 1;
            v39 = (v37 - 1) & (v68 + v39);
            v40 = (__int64 *)(v38 + 16LL * v39);
            v41 = *v40;
            if ( v33 == *v40 )
              goto LABEL_49;
            v68 = v30;
          }
        }
      }
      v45 = *(_QWORD *)(v33 + 8);
      if ( v45 )
        break;
LABEL_85:
      if ( v83 == v32 )
        goto LABEL_41;
      ++v32;
    }
    while ( 1 )
    {
      v52 = sub_1648700(v45);
      v53 = *((unsigned __int8 *)v52 + 16);
      v54 = (__int64)v52;
      if ( (unsigned __int8)v53 <= 0x17u )
        goto LABEL_65;
      v55 = *(_BYTE *)(a1 + 32) & 1;
      if ( v55 )
      {
        v46 = a1 + 40;
        v47 = 3;
      }
      else
      {
        v56 = *(unsigned int *)(a1 + 48);
        v46 = *(_QWORD *)(a1 + 40);
        if ( !(_DWORD)v56 )
          goto LABEL_88;
        v47 = v56 - 1;
      }
      v48 = v47 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v49 = v46 + 16LL * v48;
      v30 = *(_QWORD *)v49;
      if ( v54 != *(_QWORD *)v49 )
        break;
LABEL_60:
      v50 = 64;
      if ( !v55 )
        v50 = 16LL * *(unsigned int *)(a1 + 48);
      if ( v49 == v50 + v46 )
        goto LABEL_74;
      v51 = (_QWORD **)(*(_QWORD *)a1 + 176LL * *(int *)(v49 + 8));
      if ( !v51 )
        goto LABEL_74;
      if ( **v51 == v54 )
      {
        if ( v53 != 55 )
        {
          if ( v53 == 78 )
          {
            v67 = sub_14C3B40(v54, *(__int64 **)(a1 + 1328));
            if ( !sub_14C3B20(v67, 1) || v33 != *(_QWORD *)(v54 + 24 * (1LL - (*(_DWORD *)(v54 + 20) & 0xFFFFFFF))) )
              goto LABEL_65;
            goto LABEL_74;
          }
          if ( v53 != 54 )
            goto LABEL_65;
        }
        if ( v33 != *(_QWORD *)(v54 - 24) )
          goto LABEL_65;
LABEL_74:
        v57 = *(_QWORD **)(a1 + 1248);
        v58 = 8LL * *(_QWORD *)(a1 + 1256);
        v59 = &v57[(unsigned __int64)v58 / 8];
        v60 = v58 >> 5;
        v61 = v58 >> 3;
        if ( v60 > 0 )
        {
          v62 = &v57[4 * v60];
          while ( v54 != *v57 )
          {
            if ( v54 == v57[1] )
            {
              ++v57;
              goto LABEL_81;
            }
            if ( v54 == v57[2] )
            {
              v57 += 2;
              goto LABEL_81;
            }
            if ( v54 == v57[3] )
            {
              v57 += 3;
              goto LABEL_81;
            }
            v57 += 4;
            if ( v62 == v57 )
            {
              v61 = v59 - v57;
              goto LABEL_91;
            }
          }
          goto LABEL_81;
        }
LABEL_91:
        if ( v61 == 2 )
          goto LABEL_108;
        if ( v61 != 3 )
        {
          if ( v61 != 1 )
            goto LABEL_82;
          goto LABEL_94;
        }
        if ( v54 != *v57 )
        {
          ++v57;
LABEL_108:
          if ( v54 != *v57 )
          {
            ++v57;
LABEL_94:
            if ( v54 != *v57 )
              goto LABEL_82;
          }
        }
LABEL_81:
        if ( v59 != v57 )
          goto LABEL_65;
LABEL_82:
        v87.m128i_i64[0] = v33;
        v87.m128i_i64[1] = v54;
        LODWORD(v88) = v86;
        v63 = *(unsigned int *)(a1 + 392);
        if ( (unsigned int)v63 >= *(_DWORD *)(a1 + 396) )
        {
          sub_16CD150(a1 + 384, v80, 0, 24, v47, v30);
          v63 = *(unsigned int *)(a1 + 392);
        }
        v64 = (__m128i *)(*(_QWORD *)(a1 + 384) + 24 * v63);
        *v64 = _mm_loadu_si128(&v87);
        v64[1].m128i_i64[0] = v88;
        ++*(_DWORD *)(a1 + 392);
        v45 = *(_QWORD *)(v45 + 8);
        if ( !v45 )
          goto LABEL_85;
      }
      else
      {
LABEL_65:
        v45 = *(_QWORD *)(v45 + 8);
        if ( !v45 )
          goto LABEL_85;
      }
    }
    v66 = 1;
    while ( v30 != -8 )
    {
      v69 = v66 + 1;
      v48 = v47 & (v66 + v48);
      v49 = v46 + 16LL * v48;
      v30 = *(_QWORD *)v49;
      if ( v54 == *(_QWORD *)v49 )
        goto LABEL_60;
      v66 = v69;
    }
    if ( v55 )
    {
      v65 = 64;
    }
    else
    {
      v56 = *(unsigned int *)(a1 + 48);
LABEL_88:
      v65 = 16 * v56;
    }
    v49 = v46 + v65;
    goto LABEL_60;
  }
  result = (_QWORD **)(v9 + 1);
  while ( *(_QWORD *)*v9 == **result )
  {
    if ( &v9[(unsigned int)(a3 - 2) + 2] == (__int64 *)++result )
      goto LABEL_39;
  }
  return result;
}
