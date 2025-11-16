// Function: sub_2E0DC20
// Address: 0x2e0dc20
//
unsigned __int64 __fastcall sub_2E0DC20(
        __int64 *a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  unsigned __int64 v6; // r8
  __int64 v8; // r13
  _QWORD *v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rsi
  __int64 v13; // rdx
  _QWORD *v14; // r12
  __int64 v15; // rdi
  unsigned int v16; // ecx
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  _QWORD *v22; // r12
  char v23; // r14
  __m128i *v24; // rax
  unsigned __int64 result; // rax
  _QWORD *v26; // rdx
  _QWORD *v27; // rsi
  int v28; // r10d
  __int64 v29; // rax
  unsigned __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // edx
  unsigned int v34; // r13d
  unsigned int v35; // eax
  int v36; // r13d
  __int64 v37; // rax
  unsigned __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // [rsp+8h] [rbp-78h]
  int v43; // [rsp+8h] [rbp-78h]
  unsigned __int64 v44; // [rsp+10h] [rbp-70h]
  unsigned __int64 v45; // [rsp+10h] [rbp-70h]
  unsigned __int64 v48; // [rsp+20h] [rbp-60h]
  unsigned __int64 v49; // [rsp+20h] [rbp-60h]
  __int64 v50; // [rsp+28h] [rbp-58h]
  unsigned __int64 v51; // [rsp+28h] [rbp-58h]
  __m128i v52; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v53; // [rsp+40h] [rbp-40h]

  v6 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v8 = (a2 >> 1) & 3;
  v50 = *a1;
  v10 = *(_QWORD **)(*a1 + 96);
  if ( v8 == 3 )
  {
    v11 = (_QWORD *)v10[2];
    v12 = v10 + 1;
    v13 = *(_QWORD *)(v6 + 8) & 0xFFFFFFFFFFFFFFF9LL;
    if ( v11 )
      goto LABEL_3;
LABEL_21:
    v14 = v12;
    if ( v12 == (_QWORD *)v10[3] )
    {
LABEL_22:
      if ( a4 )
      {
LABEL_23:
        v52.m128i_i64[0] = a2;
        v52.m128i_i64[1] = v6 | 6;
        v53 = a4;
        v27 = sub_2E0D9D0(v10, v12, v52.m128i_i64);
        if ( v26 )
          sub_2E09B80((__int64)v10, (__int64)v27, v26, &v52);
        return a4;
      }
      v36 = *(_DWORD *)(v50 + 72);
      v37 = *a3;
      a3[10] += 16;
      v38 = (v37 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( a3[1] >= v38 + 16 && v37 )
      {
        *a3 = v38 + 16;
        a4 = (v37 + 15) & 0xFFFFFFFFFFFFFFF0LL;
        if ( !v38 )
        {
LABEL_45:
          v39 = *(unsigned int *)(v50 + 72);
          if ( v39 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 76) )
          {
            v49 = v6;
            sub_C8D5F0(v50 + 64, (const void *)(v50 + 80), v39 + 1, 8u, v6, a6);
            v39 = *(unsigned int *)(v50 + 72);
            v6 = v49;
          }
          *(_QWORD *)(*(_QWORD *)(v50 + 64) + 8 * v39) = v38;
          ++*(_DWORD *)(v50 + 72);
          v10 = *(_QWORD **)(*a1 + 96);
          v12 = v10 + 1;
          goto LABEL_23;
        }
      }
      else
      {
        v45 = v6;
        v41 = sub_9D1E70((__int64)a3, 16, 16, 4);
        v6 = v45;
        a4 = v41;
        v38 = v41;
      }
      *(_DWORD *)a4 = v36;
      *(_QWORD *)(a4 + 8) = a2;
      goto LABEL_45;
    }
    goto LABEL_10;
  }
  v11 = (_QWORD *)v10[2];
  v12 = v10 + 1;
  v13 = v6 | (2 * v8 + 2);
  if ( !v11 )
    goto LABEL_21;
LABEL_3:
  v14 = v12;
  a6 = v13 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = (v13 >> 1) & 3;
  v16 = v8 | *(_DWORD *)(v6 + 24);
  do
  {
    while ( 1 )
    {
      v17 = *(_DWORD *)((v11[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v11[4] >> 1) & 3;
      if ( v17 > v16
        || v17 >= v16
        && ((unsigned int)v15 | *(_DWORD *)(a6 + 24)) < (*(_DWORD *)((v11[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                       | (unsigned int)((__int64)v11[5] >> 1) & 3) )
      {
        break;
      }
      v11 = (_QWORD *)v11[3];
      if ( !v11 )
        goto LABEL_9;
    }
    v14 = v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v11 );
LABEL_9:
  if ( (_QWORD *)v10[3] != v14 )
  {
LABEL_10:
    v42 = v6;
    v18 = sub_220EFE0((__int64)v14);
    v6 = v42;
    if ( (*(_DWORD *)(v42 + 24) | (unsigned int)v8) < (*(_DWORD *)((*(_QWORD *)(v18 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                     | (unsigned int)(*(__int64 *)(v18 + 40) >> 1) & 3) )
      v14 = (_QWORD *)v18;
  }
  if ( v12 == v14 )
    goto LABEL_22;
  v19 = v14[4];
  if ( v6 != (v19 & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( a4 )
    {
LABEL_15:
      v52.m128i_i64[0] = a2;
      v51 = v6;
      v52.m128i_i64[1] = v6 | 6;
      v53 = a4;
      v20 = sub_2E0D9D0(v10, v14, v52.m128i_i64);
      v22 = v21;
      if ( v21 )
      {
        v23 = 1;
        if ( !v20 && v10 + 1 != v21 )
        {
          v32 = v21[4];
          v33 = *(_DWORD *)(v51 + 24);
          v34 = v33 | v8;
          v35 = *(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v32 >> 1) & 3;
          v23 = v34 < v35
             || v34 <= v35
             && (v33 | 3u) < (*(_DWORD *)((v22[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                            | (unsigned int)((__int64)v22[5] >> 1) & 3);
        }
        v24 = (__m128i *)sub_22077B0(0x38u);
        v24[2] = _mm_loadu_si128(&v52);
        v24[3].m128i_i64[0] = v53;
        sub_220F040(v23, (__int64)v24, v22, v10 + 1);
        ++v10[5];
      }
      return a4;
    }
    v28 = *(_DWORD *)(v50 + 72);
    v29 = *a3;
    a3[10] += 16;
    v30 = (v29 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( a3[1] >= v30 + 16 && v29 )
    {
      *a3 = v30 + 16;
      a4 = (v29 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v30 )
      {
LABEL_32:
        v31 = *(unsigned int *)(v50 + 72);
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 76) )
        {
          v48 = v6;
          sub_C8D5F0(v50 + 64, (const void *)(v50 + 80), v31 + 1, 8u, v6, a6);
          v6 = v48;
          v31 = *(unsigned int *)(v50 + 72);
        }
        *(_QWORD *)(*(_QWORD *)(v50 + 64) + 8 * v31) = v30;
        ++*(_DWORD *)(v50 + 72);
        v10 = *(_QWORD **)(*a1 + 96);
        goto LABEL_15;
      }
    }
    else
    {
      v43 = v28;
      v44 = v6;
      v40 = sub_9D1E70((__int64)a3, 16, 16, 4);
      v28 = v43;
      v6 = v44;
      a4 = v40;
      v30 = v40;
    }
    *(_DWORD *)a4 = v28;
    *(_QWORD *)(a4 + 8) = a2;
    goto LABEL_32;
  }
  result = v14[6];
  if ( (*(_DWORD *)(v6 + 24) | (unsigned int)(v19 >> 1) & 3) >= (*(_DWORD *)(v6 + 24) | (unsigned int)v8) && v19 != a2 )
  {
    *(_QWORD *)(result + 8) = a2;
    result = v14[6];
    v14[4] = *(_QWORD *)(result + 8);
  }
  return result;
}
