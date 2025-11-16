// Function: sub_3528EA0
// Address: 0x3528ea0
//
__int64 __fastcall sub_3528EA0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r13
  unsigned __int32 v11; // r12d
  unsigned int v12; // r14d
  __int64 v13; // rdx
  _BYTE *v14; // r15
  __int64 v15; // rax
  _BYTE *v16; // r14
  _BYTE *v17; // rbx
  unsigned int v18; // r15d
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned int v22; // eax
  _BYTE *v23; // r12
  __int64 *v24; // rbx
  __int64 v25; // r14
  __int64 v26; // r15
  __int64 v27; // r13
  __int64 v28; // rsi
  unsigned int v30; // eax
  unsigned int v31; // [rsp+Ch] [rbp-74h]
  __m128i v33; // [rsp+20h] [rbp-60h] BYREF
  __int64 v34; // [rsp+30h] [rbp-50h]
  __int64 v35; // [rsp+38h] [rbp-48h]
  __m128i v36[4]; // [rsp+40h] [rbp-40h] BYREF

  v7 = *a3;
  v8 = *((unsigned int *)a3 + 2);
  v35 = a1;
  v9 = *(_QWORD *)(v7 + 8 * v8 - 8);
  v33.m128i_i64[0] = a5;
  v33.m128i_i64[1] = a6;
  v10 = v9;
  if ( v8 == 1 )
  {
    v11 = 0;
  }
  else
  {
    v11 = 0;
    v12 = 0;
    v13 = 0;
    while ( 1 )
    {
      v11 += sub_2FF8080(a1 + 672, *(_QWORD *)(v7 + 8 * v13), 1);
      v13 = ++v12;
      if ( v12 >= (unsigned __int64)*((unsigned int *)a3 + 2) - 1 )
        break;
      v7 = *a3;
    }
  }
  v14 = *(_BYTE **)(v10 + 32);
  v15 = 5LL * (*(_DWORD *)(v10 + 40) & 0xFFFFFF);
  v36[0] = _mm_load_si128(&v33);
  v16 = &v14[8 * v15];
  if ( v14 != v16 )
  {
    while ( 1 )
    {
      v17 = v14;
      if ( sub_2DADC00(v14) )
        break;
      v14 += 40;
      if ( v16 == v14 )
        goto LABEL_24;
    }
    if ( v16 != v14 )
    {
      v33.m128i_i32[0] = v11;
      v18 = 0;
      v34 = v35 + 672;
      do
      {
        v19 = *((_DWORD *)v17 + 2);
        if ( v19 < 0 )
        {
          v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v35 + 304) + 56LL) + 16LL * (v19 & 0x7FFFFFFF) + 8)
                          + 32LL);
          if ( v20 )
          {
            v21 = *(_QWORD *)(v20 + 16);
            if ( v21 && sub_2EE9690((__int64)v36, a2, *(_QWORD *)(v20 + 16)) )
            {
              v31 = sub_2E89C70(v21, *((_DWORD *)v17 + 2), 0, 0);
              v30 = sub_2E8E710(v10, *((_DWORD *)v17 + 2), 0, 0, 0);
              v22 = sub_2FF8170(v34, v10, v30, v21, v31);
            }
            else
            {
              v22 = sub_2FF8080(v34, v10, 1);
            }
            if ( v18 < v22 )
              v18 = v22;
          }
        }
        if ( v17 + 40 == v16 )
          break;
        v23 = v17 + 40;
        while ( 1 )
        {
          v17 = v23;
          if ( sub_2DADC00(v23) )
            break;
          v23 += 40;
          if ( v16 == v23 )
            goto LABEL_23;
        }
      }
      while ( v16 != v23 );
LABEL_23:
      v11 = v18 + v33.m128i_i32[0];
    }
  }
LABEL_24:
  v24 = *(__int64 **)a4;
  v25 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  if ( v25 == *(_QWORD *)a4 )
  {
    v26 = 0;
  }
  else
  {
    LODWORD(v26) = 0;
    v27 = v35 + 672;
    do
    {
      v28 = *v24++;
      v26 = (unsigned int)sub_2FF8080(v27, v28, 1) + (unsigned int)v26;
    }
    while ( (__int64 *)v25 != v24 );
  }
  return (v26 << 32) | v11;
}
