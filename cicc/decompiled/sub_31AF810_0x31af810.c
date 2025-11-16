// Function: sub_31AF810
// Address: 0x31af810
//
__int64 __fastcall sub_31AF810(__int64 *a1, __int64 *a2, const __m128i *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  __m128i v10; // xmm1
  __int64 result; // rax
  __int64 v12; // rdx
  __m128i v13; // xmm3
  __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  __m128i v22; // xmm5
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 *v28; // rax
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 v31; // rax
  __m128i v32; // xmm7
  __m128i v33; // xmm6
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 *v39; // rdx
  __int64 v40; // r9
  __int64 v41; // rsi
  int v42; // edx
  int v43; // r10d
  int v44; // edx
  int v45; // r10d
  int v46; // eax
  int v47; // r10d
  __int64 v48; // [rsp-F0h] [rbp-F0h]
  __int64 v49; // [rsp-E8h] [rbp-E8h]
  __int64 v50; // [rsp-E0h] [rbp-E0h]
  _OWORD v51[2]; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v52; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v53; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v54; // [rsp-98h] [rbp-98h] BYREF
  __m128i v55; // [rsp-88h] [rbp-88h] BYREF
  __m128i v56; // [rsp-78h] [rbp-78h] BYREF
  __m128i v57; // [rsp-68h] [rbp-68h] BYREF
  __m128i v58; // [rsp-58h] [rbp-58h] BYREF
  __m128i v59; // [rsp-48h] [rbp-48h] BYREF

  v3 = *a2;
  v4 = *a1;
  if ( *(_DWORD *)(*(_QWORD *)(v4 + 48) + 72LL) == 2 )
    return nullsub_2036(v4, v3);
  v6 = v4;
  v7 = v3;
  v8 = sub_371B390(a3);
  v9 = a3->m128i_i64[1];
  v50 = v8;
  v48 = *(_QWORD *)(v4 + 40);
  sub_318B480((__int64)&v56, v3);
  v10 = _mm_loadu_si128(&v57);
  v58 = _mm_loadu_si128(&v56);
  v59 = v10;
  result = sub_371B2F0(&v58);
  if ( v9 != v58.m128i_i64[1] )
  {
    sub_318B480((__int64)&v58, *(_QWORD *)(v4 + 32));
    v12 = v3;
    if ( v9 != v58.m128i_i64[1] )
    {
      v12 = *(_QWORD *)(v4 + 32);
      if ( v3 == v12 )
        v12 = sub_318B4B0(v3);
    }
    v49 = v12;
    sub_318B480((__int64)&v58, *(_QWORD *)(v4 + 40));
    v13 = _mm_loadu_si128(&v59);
    v54 = _mm_loadu_si128(&v58);
    v55 = v13;
    sub_371B2F0(&v54);
    v14 = v49;
    result = v3;
    if ( v9 != v54.m128i_i64[1] )
    {
      result = *(_QWORD *)(v4 + 40);
      if ( v3 == result )
      {
        result = sub_318B520(v3);
        v14 = v49;
      }
    }
    *(_QWORD *)(v4 + 32) = v14;
    *(_QWORD *)(v4 + 40) = result;
  }
  if ( v3 )
  {
    result = *(unsigned int *)(v4 + 24);
    v15 = *(_QWORD *)(v4 + 8);
    if ( (_DWORD)result )
    {
      v16 = (result - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v17 = (__int64 *)(v15 + 16LL * (((_DWORD)result - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4))));
      v18 = *v17;
      if ( v7 == *v17 )
      {
LABEL_13:
        result = v15 + 16 * result;
        if ( v17 == (__int64 *)result )
          return result;
        v19 = v17[1];
        if ( !v19 || *(_DWORD *)(v19 + 16) != 1 )
          return result;
        v20 = *(_QWORD *)(v19 + 40);
        if ( v20 )
          *(_QWORD *)(v20 + 48) = *(_QWORD *)(v19 + 48);
        v21 = *(_QWORD *)(v19 + 48);
        if ( v21 )
          *(_QWORD *)(v21 + 40) = *(_QWORD *)(v19 + 40);
        *(_QWORD *)(v19 + 40) = 0;
        *(_QWORD *)(v19 + 48) = 0;
        if ( a3->m128i_i64[1] != *(_QWORD *)(v50 + 16) + 48LL )
        {
          sub_318B480((__int64)&v52, v48);
          v22 = _mm_loadu_si128(&v53);
          v54 = _mm_loadu_si128(&v52);
          v55 = v22;
          sub_371B2F0(&v54);
          if ( a3->m128i_i64[1] != v54.m128i_i64[1] )
          {
            v23 = sub_371B3B0(a3, a3->m128i_i64[1], a3[1].m128i_i64[0]);
            v24 = *(unsigned int *)(v6 + 24);
            v25 = *(_QWORD *)(v6 + 8);
            v26 = v23;
            if ( (_DWORD)v24 )
            {
              v27 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v28 = (__int64 *)(v25 + 16LL * v27);
              v29 = *v28;
              if ( v26 == *v28 )
              {
LABEL_24:
                if ( v28 != (__int64 *)(v25 + 16 * v24) )
                {
                  v30 = v28[1];
LABEL_26:
                  v31 = sub_31B9B30(v6, v30, 0, v19);
                  *(_QWORD *)(v19 + 40) = v31;
                  if ( v31 )
                    *(_QWORD *)(v31 + 48) = v19;
                  result = sub_31B9BF0(v6, v30, 1, v19);
                  *(_QWORD *)(v19 + 48) = result;
                  if ( result )
                    *(_QWORD *)(result + 40) = v19;
                  return result;
                }
              }
              else
              {
                v46 = 1;
                while ( v29 != -4096 )
                {
                  v47 = v46 + 1;
                  v27 = (v24 - 1) & (v46 + v27);
                  v28 = (__int64 *)(v25 + 16LL * v27);
                  v29 = *v28;
                  if ( v26 == *v28 )
                    goto LABEL_24;
                  v46 = v47;
                }
              }
            }
            v30 = 0;
            goto LABEL_26;
          }
        }
        v32 = _mm_loadu_si128(a3 + 1);
        v54 = _mm_loadu_si128(a3);
        v55 = v32;
        sub_371B3D0(&v54);
        v33 = _mm_loadu_si128(&v54);
        v51[1] = _mm_loadu_si128(&v55);
        v51[0] = v33;
        v34 = sub_371B3B0(v51, v33.m128i_i64[1], v55.m128i_i64[0]);
        v35 = *(_QWORD *)(v6 + 8);
        v36 = v34;
        v37 = *(unsigned int *)(v6 + 24);
        if ( (_DWORD)v37 )
        {
          v38 = (v37 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v39 = (__int64 *)(v35 + 16LL * v38);
          v40 = *v39;
          if ( v36 == *v39 )
          {
LABEL_33:
            if ( v39 != (__int64 *)(v35 + 16 * v37) )
            {
              v41 = v39[1];
              goto LABEL_35;
            }
          }
          else
          {
            v44 = 1;
            while ( v40 != -4096 )
            {
              v45 = v44 + 1;
              v38 = (v37 - 1) & (v44 + v38);
              v39 = (__int64 *)(v35 + 16LL * v38);
              v40 = *v39;
              if ( v36 == *v39 )
                goto LABEL_33;
              v44 = v45;
            }
          }
        }
        v41 = 0;
LABEL_35:
        result = sub_31B9B30(v6, v41, 1, v19);
        *(_QWORD *)(v19 + 40) = result;
        if ( result )
          *(_QWORD *)(result + 48) = v19;
        return result;
      }
      v42 = 1;
      while ( v18 != -4096 )
      {
        v43 = v42 + 1;
        v16 = (result - 1) & (v42 + v16);
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v7 == *v17 )
          goto LABEL_13;
        v42 = v43;
      }
    }
  }
  return result;
}
