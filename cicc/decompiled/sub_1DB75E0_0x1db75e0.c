// Function: sub_1DB75E0
// Address: 0x1db75e0
//
__int64 __fastcall sub_1DB75E0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned __int64 v4; // r8
  __int64 v6; // r13
  _QWORD *v8; // r15
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  _QWORD *v12; // r12
  unsigned __int64 v13; // r9
  __int64 v14; // rdi
  unsigned int v15; // ecx
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r12
  _BOOL4 v22; // r14d
  __m128i *v23; // rax
  __int64 result; // rax
  __int64 v25; // rdx
  _QWORD *v26; // rsi
  int v27; // r15d
  __int64 v28; // rax
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // edx
  unsigned int v33; // r13d
  unsigned int v34; // eax
  int v35; // r12d
  __int64 v36; // rax
  int v37; // r9d
  __int64 v38; // rax
  unsigned __int64 v39; // [rsp+8h] [rbp-78h]
  unsigned __int64 v40; // [rsp+10h] [rbp-70h]
  unsigned __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-58h]
  unsigned __int64 v45; // [rsp+28h] [rbp-58h]
  __m128i v46; // [rsp+30h] [rbp-50h] BYREF
  __int64 v47; // [rsp+40h] [rbp-40h]

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (a2 >> 1) & 3;
  v44 = *a1;
  v8 = *(_QWORD **)(*a1 + 96);
  if ( v6 == 3 )
  {
    v9 = (_QWORD *)v8[2];
    v10 = v8 + 1;
    v11 = *(_QWORD *)(v4 + 8) & 0xFFFFFFFFFFFFFFF9LL;
    if ( v9 )
      goto LABEL_3;
LABEL_21:
    v12 = v10;
    if ( v10 == (_QWORD *)v8[3] )
      goto LABEL_22;
    goto LABEL_10;
  }
  v9 = (_QWORD *)v8[2];
  v10 = v8 + 1;
  v11 = v4 | (2 * v6 + 2);
  if ( !v9 )
    goto LABEL_21;
LABEL_3:
  v12 = v10;
  v13 = v11 & 0xFFFFFFFFFFFFFFF8LL;
  v14 = (v11 >> 1) & 3;
  v15 = v6 | *(_DWORD *)(v4 + 24);
  do
  {
    while ( 1 )
    {
      v16 = *(_DWORD *)((v9[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v9[4] >> 1) & 3;
      if ( v16 > v15
        || v16 >= v15
        && ((unsigned int)v14 | *(_DWORD *)(v13 + 24)) < (*(_DWORD *)((v9[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                        | (unsigned int)((__int64)v9[5] >> 1) & 3) )
      {
        break;
      }
      v9 = (_QWORD *)v9[3];
      if ( !v9 )
        goto LABEL_9;
    }
    v12 = v9;
    v9 = (_QWORD *)v9[2];
  }
  while ( v9 );
LABEL_9:
  if ( (_QWORD *)v8[3] != v12 )
  {
LABEL_10:
    v39 = v4;
    v17 = sub_220EFE0(v12);
    v4 = v39;
    if ( (*(_DWORD *)(v39 + 24) | (unsigned int)v6) < (*(_DWORD *)((*(_QWORD *)(v17 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                     | (unsigned int)(*(__int64 *)(v17 + 40) >> 1) & 3) )
      v12 = (_QWORD *)v17;
  }
  if ( v10 == v12 )
  {
LABEL_22:
    if ( !a4 )
    {
      v41 = v4;
      v35 = *(_DWORD *)(v44 + 72);
      v36 = sub_145CDC0(0x10u, a3);
      v4 = v41;
      a4 = v36;
      if ( v36 )
      {
        *(_DWORD *)v36 = v35;
        *(_QWORD *)(v36 + 8) = a2;
      }
      v38 = *(unsigned int *)(v44 + 72);
      if ( (unsigned int)v38 >= *(_DWORD *)(v44 + 76) )
      {
        sub_16CD150(v44 + 64, (const void *)(v44 + 80), 0, 8, v41, v37);
        v38 = *(unsigned int *)(v44 + 72);
        v4 = v41;
      }
      *(_QWORD *)(*(_QWORD *)(v44 + 64) + 8 * v38) = a4;
      ++*(_DWORD *)(v44 + 72);
      v8 = *(_QWORD **)(*a1 + 96);
      v10 = v8 + 1;
    }
    v46.m128i_i64[0] = a2;
    v46.m128i_i64[1] = v4 | 6;
    v47 = a4;
    v26 = sub_1DB7390(v8, v10, v46.m128i_i64);
    if ( v25 )
      sub_1DB3B70((__int64)v8, (__int64)v26, v25, &v46);
    return a4;
  }
  v18 = v12[4];
  if ( v4 != (v18 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !a4 )
    {
      v40 = v4;
      v27 = *(_DWORD *)(v44 + 72);
      v28 = sub_145CDC0(0x10u, a3);
      v4 = v40;
      a4 = v28;
      if ( v28 )
      {
        *(_DWORD *)v28 = v27;
        *(_QWORD *)(v28 + 8) = a2;
      }
      v30 = *(unsigned int *)(v44 + 72);
      if ( (unsigned int)v30 >= *(_DWORD *)(v44 + 76) )
      {
        sub_16CD150(v44 + 64, (const void *)(v44 + 80), 0, 8, v40, v29);
        v30 = *(unsigned int *)(v44 + 72);
        v4 = v40;
      }
      *(_QWORD *)(*(_QWORD *)(v44 + 64) + 8 * v30) = a4;
      ++*(_DWORD *)(v44 + 72);
      v8 = *(_QWORD **)(*a1 + 96);
    }
    v46.m128i_i64[0] = a2;
    v45 = v4;
    v46.m128i_i64[1] = v4 | 6;
    v47 = a4;
    v19 = sub_1DB7390(v8, v12, v46.m128i_i64);
    v21 = v20;
    if ( v20 )
    {
      v22 = 1;
      if ( !v19 && v20 != v8 + 1 )
      {
        v31 = v20[4];
        v32 = *(_DWORD *)(v45 + 24);
        v33 = v32 | v6;
        v34 = *(_DWORD *)((v31 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v31 >> 1) & 3;
        v22 = v33 < v34
           || v33 <= v34
           && (v32 | 3u) < (*(_DWORD *)((v21[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v21[5] >> 1) & 3);
      }
      v23 = (__m128i *)sub_22077B0(56);
      v23[2] = _mm_loadu_si128(&v46);
      v23[3].m128i_i64[0] = v47;
      sub_220F040(v22, v23, v21, v8 + 1);
      ++v8[5];
    }
    return a4;
  }
  result = v12[6];
  if ( (*(_DWORD *)(v4 + 24) | (unsigned int)(v18 >> 1) & 3) >= (*(_DWORD *)(v4 + 24) | (unsigned int)v6) && v18 != a2 )
  {
    *(_QWORD *)(result + 8) = a2;
    result = v12[6];
    v12[4] = *(_QWORD *)(result + 8);
  }
  return result;
}
