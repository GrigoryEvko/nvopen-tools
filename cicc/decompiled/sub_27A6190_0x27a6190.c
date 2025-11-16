// Function: sub_27A6190
// Address: 0x27a6190
//
__int64 __fastcall sub_27A6190(__int64 a1, const __m128i *a2, const __m128i *a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v8; // rdi
  unsigned __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // rbx
  __int64 result; // rax
  const __m128i *v13; // r15
  __int64 v15; // r12
  __m128i v16; // xmm1
  __m128i v17; // xmm0
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rcx
  int v23; // esi
  __int64 v24; // rdi
  int v25; // esi
  unsigned int v26; // ecx
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r8
  __int64 v31; // rdi
  const __m128i *v32; // r9
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __m128i *v35; // rax
  const __m128i *v36; // rcx
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // r9
  __m128i *v40; // rax
  unsigned int v41; // r8d
  const void *v42; // rsi
  unsigned __int64 v43; // rdx
  char *v44; // [rsp+0h] [rbp-90h]
  char *v45; // [rsp+0h] [rbp-90h]
  const void *v46; // [rsp+8h] [rbp-88h]
  int v49; // [rsp+3Ch] [rbp-54h] BYREF
  _OWORD v50[5]; // [rsp+40h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(a4 + 48);
  v49 = qword_4FFC3E8;
  v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == a4 + 48 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = v9 - 24;
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  result = a6 + 16;
  v46 = (const void *)(a6 + 16);
  if ( a3 != a2 )
  {
    result = a1;
    v13 = a2;
    v15 = result;
    while ( 1 )
    {
      v16 = _mm_loadu_si128(v13 + 1);
      v17 = _mm_loadu_si128(v13);
      v50[1] = v16;
      v50[0] = v17;
      if ( !v16.m128i_i64[1] )
        goto LABEL_23;
      if ( !*(_QWORD *)(v11 + 16) )
        goto LABEL_18;
      v18 = 32LL * (*(_DWORD *)(v16.m128i_i64[1] + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v16.m128i_i64[1] + 7) & 0x40) != 0 )
      {
        result = *(_QWORD *)(v16.m128i_i64[1] - 8);
        v19 = v18 >> 5;
        v20 = result + v18;
        v21 = v18 >> 7;
        if ( v21 )
          goto LABEL_11;
      }
      else
      {
        v20 = v16.m128i_i64[1];
        result = v16.m128i_i64[1] - v18;
        v19 = v18 >> 5;
        v21 = v18 >> 7;
        if ( v21 )
        {
LABEL_11:
          v22 = result + (v21 << 7);
          while ( v11 != *(_QWORD *)result )
          {
            if ( v11 == *(_QWORD *)(result + 32) )
            {
              result += 32;
              break;
            }
            if ( v11 == *(_QWORD *)(result + 64) )
            {
              result += 64;
              break;
            }
            if ( v11 == *(_QWORD *)(result + 96) )
            {
              result += 96;
              break;
            }
            result += 128;
            if ( v22 == result )
            {
              v19 = (v20 - result) >> 5;
              if ( v19 != 2 )
                goto LABEL_27;
              goto LABEL_38;
            }
          }
LABEL_17:
          if ( result != v20 )
            goto LABEL_23;
LABEL_18:
          if ( a5 != 1 )
            goto LABEL_19;
          goto LABEL_31;
        }
      }
      if ( v19 == 2 )
        goto LABEL_38;
LABEL_27:
      if ( v19 == 3 )
        break;
      if ( v19 != 1 )
        goto LABEL_18;
      if ( v11 == *(_QWORD *)result )
        goto LABEL_17;
LABEL_30:
      if ( a5 != 1 )
      {
LABEL_19:
        result = *(_QWORD *)(v15 + 248);
        v23 = *(_DWORD *)(result + 56);
        v24 = *(_QWORD *)(result + 40);
        if ( v23 )
        {
          v25 = v23 - 1;
          v26 = v25 & (((unsigned __int32)v16.m128i_i32[2] >> 9) ^ ((unsigned __int32)v16.m128i_i32[2] >> 4));
          result = v24 + 16LL * v26;
          v27 = *(_QWORD *)result;
          if ( v16.m128i_i64[1] == *(_QWORD *)result )
          {
LABEL_21:
            v28 = *(_QWORD *)(result + 8);
            if ( v28 )
            {
              result = sub_27A5F90(v15, v11, v16.m128i_i64[1], v28, a5, &v49);
              if ( (_BYTE)result )
              {
                v36 = (const __m128i *)v50;
                v37 = *(unsigned int *)(a6 + 8);
                v38 = *(_QWORD *)a6;
                v39 = v37 + 1;
                if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
                {
                  if ( v38 > (unsigned __int64)v50 || (unsigned __int64)v50 >= v38 + 32 * v37 )
                  {
                    sub_C8D5F0(a6, v46, v39, 0x20u, v29, v39);
                    v36 = (const __m128i *)v50;
                    v38 = *(_QWORD *)a6;
                    v37 = *(unsigned int *)(a6 + 8);
                  }
                  else
                  {
                    v44 = (char *)v50 - v38;
                    sub_C8D5F0(a6, v46, v39, 0x20u, v29, v39);
                    v38 = *(_QWORD *)a6;
                    v37 = *(unsigned int *)(a6 + 8);
                    v36 = (const __m128i *)&v44[*(_QWORD *)a6];
                  }
                }
                v40 = (__m128i *)(v38 + 32 * v37);
                *v40 = _mm_loadu_si128(v36);
                v40[1] = _mm_loadu_si128(v36 + 1);
                result = a6;
                ++*(_DWORD *)(a6 + 8);
              }
            }
          }
          else
          {
            result = 1;
            while ( v27 != -4096 )
            {
              v41 = result + 1;
              v26 = v25 & (result + v26);
              result = v24 + 16LL * v26;
              v27 = *(_QWORD *)result;
              if ( v16.m128i_i64[1] == *(_QWORD *)result )
                goto LABEL_21;
              result = v41;
            }
          }
        }
        goto LABEL_23;
      }
LABEL_31:
      result = sub_27A5AB0(v15, a4, *(_QWORD *)(v16.m128i_i64[1] + 40), &v49);
      if ( !(_BYTE)result )
      {
        v31 = a6;
        v32 = (const __m128i *)v50;
        v33 = *(unsigned int *)(a6 + 8);
        v34 = *(_QWORD *)a6;
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
        {
          if ( v34 > (unsigned __int64)v50 )
          {
            v42 = v46;
            v43 = v33 + 1;
          }
          else
          {
            if ( (unsigned __int64)v50 < v34 + 32 * v33 )
            {
              v45 = (char *)v50 - v34;
              sub_C8D5F0(a6, v46, v33 + 1, 0x20u, v30, (__int64)v50);
              v34 = *(_QWORD *)a6;
              v33 = *(unsigned int *)(a6 + 8);
              v32 = (const __m128i *)&v45[*(_QWORD *)a6];
              goto LABEL_33;
            }
            v42 = v46;
            v31 = a6;
            v43 = v33 + 1;
          }
          sub_C8D5F0(v31, v42, v43, 0x20u, v30, (__int64)v50);
          v32 = (const __m128i *)v50;
          v34 = *(_QWORD *)a6;
          v33 = *(unsigned int *)(a6 + 8);
        }
LABEL_33:
        v35 = (__m128i *)(v34 + 32 * v33);
        *v35 = _mm_loadu_si128(v32);
        v35[1] = _mm_loadu_si128(v32 + 1);
        result = a6;
        ++*(_DWORD *)(a6 + 8);
      }
LABEL_23:
      v13 += 2;
      if ( a3 == v13 )
        return result;
    }
    if ( v11 == *(_QWORD *)result )
      goto LABEL_17;
    result += 32;
LABEL_38:
    if ( v11 == *(_QWORD *)result )
      goto LABEL_17;
    result += 32;
    if ( v11 == *(_QWORD *)result )
      goto LABEL_17;
    goto LABEL_30;
  }
  return result;
}
