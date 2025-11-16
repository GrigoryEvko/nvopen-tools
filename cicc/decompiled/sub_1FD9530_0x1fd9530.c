// Function: sub_1FD9530
// Address: 0x1fd9530
//
__int64 __fastcall sub_1FD9530(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  int v15; // r15d
  __int64 v16; // rax
  __m128i v17; // xmm1
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rdx
  __m128i v23; // xmm3
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  char v27; // al
  __int32 v28; // eax
  int v29; // r9d
  __int64 v30; // rax
  __m128i v31; // xmm5
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __m128i v38; // xmm7
  __m128i v39; // xmm6
  __m128i *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __m128i v43; // xmm7
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rsi
  unsigned int v49; // ecx
  __int64 *v50; // rdx
  __int64 v51; // r10
  __int32 v52; // eax
  __int64 v53; // r9
  int v54; // edx
  int v55; // r11d
  __int64 v56; // [rsp+0h] [rbp-70h]
  int v57; // [rsp+8h] [rbp-68h]
  const void *v58; // [rsp+8h] [rbp-68h]
  __m128i v59; // [rsp+10h] [rbp-60h] BYREF
  __m128i v60; // [rsp+20h] [rbp-50h] BYREF
  __int64 v61; // [rsp+30h] [rbp-40h]

  v57 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( *(char *)(a3 + 23) >= 0 )
    goto LABEL_8;
  v8 = sub_1648A40(a3);
  v10 = v8 + v9;
  if ( *(char *)(a3 + 23) >= 0 )
  {
    if ( (unsigned int)(v10 >> 4) )
LABEL_46:
      BUG();
LABEL_8:
    v14 = 0;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v10 - sub_1648A40(a3)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a3 + 23) >= 0 )
    goto LABEL_46;
  v11 = *(_DWORD *)(sub_1648A40(a3) + 8);
  if ( *(char *)(a3 + 23) >= 0 )
    BUG();
  v12 = sub_1648A40(a3);
  v14 = *(_DWORD *)(v12 + v13 - 4) - v11;
LABEL_9:
  v15 = v57 - 1 - v14;
  v58 = (const void *)(a2 + 16);
  if ( a4 == v15 )
    return 1;
  while ( 1 )
  {
    while ( 1 )
    {
      v26 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
      v53 = *(_QWORD *)(a3 + 24 * (a4 - v26));
      v27 = *(_BYTE *)(v53 + 16);
      if ( v27 == 13 )
      {
        v59.m128i_i64[0] = 1;
        v16 = *(unsigned int *)(a2 + 8);
        v60.m128i_i64[0] = 0;
        v60.m128i_i64[1] = 2;
        if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 12) )
        {
          v56 = v53;
          sub_16CD150(a2, v58, 0, 40, a5, v53);
          v16 = *(unsigned int *)(a2 + 8);
          v53 = v56;
        }
        v17 = _mm_loadu_si128(&v60);
        v18 = *(_QWORD *)a2 + 40 * v16;
        v19 = v61;
        *(__m128i *)v18 = _mm_loadu_si128(&v59);
        *(_QWORD *)(v18 + 32) = v19;
        *(__m128i *)(v18 + 16) = v17;
        v20 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v20;
        v21 = *(_DWORD *)(v53 + 32);
        if ( v21 > 0x40 )
          v22 = **(_QWORD **)(v53 + 24);
        else
          v22 = (__int64)(*(_QWORD *)(v53 + 24) << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
        v59.m128i_i64[0] = 1;
        v60.m128i_i64[0] = 0;
        v60.m128i_i64[1] = v22;
        if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, v58, 0, 40, a5, v53);
          v20 = *(unsigned int *)(a2 + 8);
        }
        v23 = _mm_loadu_si128(&v60);
        v24 = *(_QWORD *)a2 + 40 * v20;
        v25 = v61;
        *(__m128i *)v24 = _mm_loadu_si128(&v59);
        *(_QWORD *)(v24 + 32) = v25;
        *(__m128i *)(v24 + 16) = v23;
        ++*(_DWORD *)(a2 + 8);
        goto LABEL_18;
      }
      if ( v27 != 15 )
        break;
      v59.m128i_i64[0] = 1;
      v35 = *(unsigned int *)(a2 + 8);
      v60.m128i_i64[0] = 0;
      v60.m128i_i64[1] = 2;
      if ( (unsigned int)v35 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, v58, 0, 40, a5, v53);
        v35 = *(unsigned int *)(a2 + 8);
      }
      v36 = 5 * v35;
      v37 = *(_QWORD *)a2;
      v38 = _mm_loadu_si128(&v60);
      v39 = _mm_loadu_si128(&v59);
      v60.m128i_i64[0] = 0;
      v40 = (__m128i *)(v37 + 8 * v36);
      v41 = v61;
      v59.m128i_i64[0] = 1;
      *v40 = v39;
      v40[2].m128i_i64[0] = v41;
      v40[1] = v38;
      LODWORD(v40) = *(_DWORD *)(a2 + 8);
      v60.m128i_i64[1] = 0;
      v42 = (unsigned int)((_DWORD)v40 + 1);
      *(_DWORD *)(a2 + 8) = v42;
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v42 )
      {
        sub_16CD150(a2, v58, 0, 40, a5, v53);
        v42 = *(unsigned int *)(a2 + 8);
      }
LABEL_32:
      v43 = _mm_loadu_si128(&v60);
      v44 = *(_QWORD *)a2 + 40 * v42;
      v45 = v61;
      *(__m128i *)v44 = _mm_loadu_si128(&v59);
      *(_QWORD *)(v44 + 32) = v45;
      *(__m128i *)(v44 + 16) = v43;
      ++*(_DWORD *)(a2 + 8);
LABEL_18:
      if ( v15 == ++a4 )
        return 1;
    }
    if ( v27 == 53 )
      break;
    v28 = sub_1FD8F60(a1, *(_QWORD *)(a3 + 24 * (a4 - v26)));
    if ( !v28 )
      return 0;
    v59.m128i_i64[0] = 0;
    v59.m128i_i32[2] = v28;
    v30 = *(unsigned int *)(a2 + 8);
    v60 = 0u;
    v61 = 0;
    if ( (unsigned int)v30 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, v58, 0, 40, a5, v29);
      v30 = *(unsigned int *)(a2 + 8);
    }
    ++a4;
    v31 = _mm_loadu_si128(&v60);
    v32 = *(_QWORD *)a2 + 40 * v30;
    v33 = v61;
    *(__m128i *)v32 = _mm_loadu_si128(&v59);
    *(_QWORD *)(v32 + 32) = v33;
    *(__m128i *)(v32 + 16) = v31;
    ++*(_DWORD *)(a2 + 8);
    if ( v15 == a4 )
      return 1;
  }
  v46 = a1[5];
  v47 = *(unsigned int *)(v46 + 360);
  if ( !(_DWORD)v47 )
    return 0;
  v48 = *(_QWORD *)(v46 + 344);
  v49 = (v47 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
  v50 = (__int64 *)(v48 + 16LL * v49);
  v51 = *v50;
  if ( v53 == *v50 )
  {
LABEL_35:
    if ( v50 == (__int64 *)(v48 + 16 * v47) )
      return 0;
    v52 = *((_DWORD *)v50 + 2);
    v59.m128i_i64[0] = 5;
    v60.m128i_i64[0] = 0;
    v60.m128i_i32[2] = v52;
    v42 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v42 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, v58, 0, 40, a5, v53);
      v42 = *(unsigned int *)(a2 + 8);
    }
    goto LABEL_32;
  }
  v54 = 1;
  while ( v51 != -8 )
  {
    v55 = v54 + 1;
    v49 = (v47 - 1) & (v54 + v49);
    v50 = (__int64 *)(v48 + 16LL * v49);
    v51 = *v50;
    if ( v53 == *v50 )
      goto LABEL_35;
    v54 = v55;
  }
  return 0;
}
