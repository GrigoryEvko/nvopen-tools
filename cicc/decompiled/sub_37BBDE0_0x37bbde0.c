// Function: sub_37BBDE0
// Address: 0x37bbde0
//
__int64 *__fastcall sub_37BBDE0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, int a5, _QWORD *a6)
{
  __int64 v8; // rax
  char v9; // cl
  __int64 v10; // r8
  int v11; // edi
  unsigned int v12; // edx
  int *v13; // r9
  int v14; // ebx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 *result; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // r12
  __int64 *v23; // rbx
  __int64 v24; // r9
  char v25; // al
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  __int64 v29; // rax
  __m128i v30; // xmm3
  __int64 v31; // rax
  const __m128i *v32; // r12
  __int64 v33; // rdi
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __m128i *v37; // rdx
  const void *v38; // rsi
  char *v39; // r12
  __int64 v40; // rdi
  int v41; // r9d
  int v42; // r12d
  __int64 v43; // [rsp+8h] [rbp-A8h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  __int64 v47; // [rsp+28h] [rbp-88h]
  __int64 v48; // [rsp+28h] [rbp-88h]
  int v49; // [rsp+30h] [rbp-80h] BYREF
  __m128i v50; // [rsp+38h] [rbp-78h]
  __m128i v51; // [rsp+48h] [rbp-68h]
  __m128i v52; // [rsp+58h] [rbp-58h]
  __m128i v53; // [rsp+68h] [rbp-48h]

  v8 = *a4 + 856LL * *(int *)(a3 + 24);
  v9 = *(_BYTE *)(v8 + 16) & 1;
  if ( v9 )
  {
    v10 = v8 + 24;
    v11 = 7;
  }
  else
  {
    v19 = *(unsigned int *)(v8 + 32);
    v10 = *(_QWORD *)(v8 + 24);
    if ( !(_DWORD)v19 )
      goto LABEL_35;
    v11 = v19 - 1;
  }
  v12 = v11 & (37 * a5);
  v13 = (int *)(v10 + 8LL * v12);
  v14 = *v13;
  if ( *v13 == a5 )
    goto LABEL_4;
  v41 = 1;
  while ( v14 != -1 )
  {
    v42 = v41 + 1;
    v12 = v11 & (v41 + v12);
    v13 = (int *)(v10 + 8LL * v12);
    v14 = *v13;
    if ( *v13 == a5 )
      goto LABEL_4;
    v41 = v42;
  }
  if ( v9 )
  {
    v40 = 64;
    goto LABEL_36;
  }
  v19 = *(unsigned int *)(v8 + 32);
LABEL_35:
  v40 = 8 * v19;
LABEL_36:
  v13 = (int *)(v10 + v40);
LABEL_4:
  v15 = 64;
  if ( !v9 )
    v15 = 8LL * *(unsigned int *)(v8 + 32);
  v16 = *(_QWORD *)(v8 + 88);
  if ( v13 == (int *)(v10 + v15) )
    v17 = v16 + 72LL * *(unsigned int *)(v8 + 96);
  else
    v17 = v16 + 72LL * (unsigned int)v13[1];
  result = (__int64 *)*(unsigned int *)(v17 + 64);
  if ( (_DWORD)result )
  {
    result = *(__int64 **)(a2 + 8);
    v20 = *(_BYTE *)(a2 + 28) ? *(unsigned int *)(a2 + 20) : *(unsigned int *)(a2 + 16);
    v21 = (__int64)&result[v20];
    if ( result != (__int64 *)v21 )
    {
      while ( 1 )
      {
        v22 = *result;
        v23 = result;
        if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (__int64 *)v21 == ++result )
          return result;
      }
      while ( result != (__int64 *)v21 )
      {
        v46 = v21;
        sub_2E6D2E0(*(_QWORD *)(a1 + 8), a3, v22);
        v21 = v46;
        if ( v25 )
        {
          v26 = _mm_loadu_si128((const __m128i *)(v17 + 8));
          v27 = _mm_loadu_si128((const __m128i *)(v17 + 24));
          v28 = _mm_loadu_si128((const __m128i *)(v17 + 40));
          v29 = 37LL * *(int *)(v22 + 24);
          v30 = _mm_loadu_si128((const __m128i *)(v17 + 56));
          v49 = a5;
          v31 = *a6 + 16 * v29;
          v32 = (const __m128i *)&v49;
          v50 = v26;
          v33 = *(unsigned int *)(v31 + 8);
          v34 = *(unsigned int *)(v31 + 12);
          v51 = v27;
          v35 = *(_QWORD *)v31;
          v52 = v28;
          v36 = v33 + 1;
          v53 = v30;
          if ( v33 + 1 > v34 )
          {
            v43 = v46;
            v38 = (const void *)(v31 + 16);
            if ( v35 > (unsigned __int64)&v49 || (unsigned __int64)&v49 >= v35 + 72 * v33 )
            {
              v48 = v31;
              sub_C8D5F0(v31, v38, v36, 0x48u, v21, v24);
              v31 = v48;
              v32 = (const __m128i *)&v49;
              v21 = v43;
              v35 = *(_QWORD *)v48;
              v33 = *(unsigned int *)(v48 + 8);
            }
            else
            {
              v47 = v31;
              v39 = (char *)&v49 - v35;
              sub_C8D5F0(v31, v38, v36, 0x48u, v21, v24);
              v31 = v47;
              v21 = v43;
              v35 = *(_QWORD *)v47;
              v33 = *(unsigned int *)(v47 + 8);
              v32 = (const __m128i *)&v39[*(_QWORD *)v47];
            }
          }
          v37 = (__m128i *)(v35 + 72 * v33);
          *v37 = _mm_loadu_si128(v32);
          v37[1] = _mm_loadu_si128(v32 + 1);
          v37[2] = _mm_loadu_si128(v32 + 2);
          v37[3] = _mm_loadu_si128(v32 + 3);
          v37[4].m128i_i64[0] = v32[4].m128i_i64[0];
          ++*(_DWORD *)(v31 + 8);
        }
        result = v23 + 1;
        if ( v23 + 1 == (__int64 *)v21 )
          break;
        while ( 1 )
        {
          v22 = *result;
          v23 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v21 == ++result )
            return result;
        }
      }
    }
  }
  return result;
}
