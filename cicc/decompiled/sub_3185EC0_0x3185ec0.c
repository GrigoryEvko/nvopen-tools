// Function: sub_3185EC0
// Address: 0x3185ec0
//
__int64 __fastcall sub_3185EC0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  int v6; // edi
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rdx
  char *v12; // r14
  char *v13; // r8
  __int64 v14; // r9
  unsigned int v15; // r10d
  __int64 *v16; // rcx
  __int64 v17; // r11
  __int64 v18; // rcx
  _QWORD *v19; // rbx
  __int64 v20; // r8
  unsigned __int64 v21; // r12
  __m128i v22; // xmm0
  __int64 v23; // rcx
  __m128i v24; // xmm2
  __int64 v25; // rdx
  __m128i v26; // xmm0
  void (__fastcall *v27)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // rax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  void (__fastcall *v31)(__int64, __int64, __int64); // rcx
  unsigned __int64 v32; // r14
  _QWORD *v33; // rcx
  _QWORD *v34; // rdx
  unsigned __int64 v35; // rsi
  int v36; // ecx
  int v37; // r10d
  int v38; // ecx
  int v39; // ebx
  __m128i v40; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v41)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp-48h] [rbp-48h]
  __int64 v42; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)result )
  {
    v6 = result - 1;
    v7 = *a2;
    v8 = (result - 1) & (((0xBF58476D1CE4E5B9LL * *a2) >> 31) ^ (484763065 * *(_DWORD *)a2));
    v9 = (__int64 *)(v4 + 16LL * v8);
    v10 = *v9;
    if ( v7 == *v9 )
    {
LABEL_3:
      result = v4 + 16 * result;
      if ( v9 != (__int64 *)result )
      {
        v11 = *(_QWORD *)(a1 + 32);
        v12 = (char *)(v11 + 40LL * *((unsigned int *)v9 + 2));
        result = *(unsigned int *)(a1 + 40);
        v13 = (char *)(v11 + 40 * result);
        if ( v13 != v12 )
        {
          v14 = *(_QWORD *)v12;
          v15 = v6 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v12) >> 31) ^ (484763065 * *(_DWORD *)v12));
          v16 = (__int64 *)(v4 + 16LL * v15);
          v17 = *v16;
          if ( *v16 == *(_QWORD *)v12 )
          {
LABEL_6:
            *v16 = -2;
            v18 = *(unsigned int *)(a1 + 40);
            --*(_DWORD *)(a1 + 16);
            v11 = *(_QWORD *)(a1 + 32);
            ++*(_DWORD *)(a1 + 20);
            LODWORD(result) = v18;
            v13 = (char *)(v11 + 40 * v18);
          }
          else
          {
            v38 = 1;
            while ( v17 != -1 )
            {
              v39 = v38 + 1;
              v15 = v6 & (v38 + v15);
              v16 = (__int64 *)(v4 + 16LL * v15);
              v17 = *v16;
              if ( v14 == *v16 )
                goto LABEL_6;
              v38 = v39;
            }
          }
          v19 = v12 + 40;
          v20 = v13 - (v12 + 40);
          v21 = 0xCCCCCCCCCCCCCCCDLL * (v20 >> 3);
          if ( v20 > 0 )
          {
            do
            {
              v22 = _mm_loadu_si128((const __m128i *)(v19 + 1));
              *(v19 - 5) = *v19;
              *(__m128i *)(v19 + 1) = _mm_loadu_si128(&v40);
              v40 = v22;
              v23 = v19[3];
              v24 = _mm_loadu_si128((const __m128i *)v19 - 2);
              v19[3] = 0;
              v25 = v19[4];
              v19[4] = v42;
              v26 = _mm_loadu_si128(&v40);
              v40 = v24;
              v27 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))*(v19 - 2);
              *((__m128i *)v19 - 2) = v26;
              v41 = v27;
              *(v19 - 2) = v23;
              v28 = *(v19 - 1);
              v42 = v28;
              *(v19 - 1) = v25;
              if ( v27 )
                v27(&v40, &v40, 3, v28, v20, v14, v40.m128i_i64[0], v40.m128i_i64[1], v41);
              v19 += 5;
              --v21;
            }
            while ( v21 );
            LODWORD(result) = *(_DWORD *)(a1 + 40);
            v11 = *(_QWORD *)(a1 + 32);
          }
          v29 = (unsigned int)(result - 1);
          *(_DWORD *)(a1 + 40) = v29;
          v30 = v11 + 40 * v29;
          v31 = *(void (__fastcall **)(__int64, __int64, __int64))(v30 + 24);
          if ( v31 )
          {
            v31(v30 + 8, v30 + 8, 3);
            v11 = *(_QWORD *)(a1 + 32);
          }
          result = v11 + 40LL * *(unsigned int *)(a1 + 40);
          if ( v12 != (char *)result )
          {
            v32 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v12[-v11] >> 3);
            result = *(unsigned int *)(a1 + 16);
            if ( (_DWORD)result )
            {
              v33 = *(_QWORD **)(a1 + 8);
              v34 = &v33[2 * *(unsigned int *)(a1 + 24)];
              if ( v33 != v34 )
              {
                while ( 1 )
                {
                  result = (__int64)v33;
                  if ( *v33 <= 0xFFFFFFFFFFFFFFFDLL )
                    break;
                  v33 += 2;
                  if ( v34 == v33 )
                    return result;
                }
                if ( v33 != v34 )
                {
                  do
                  {
                    v35 = *(unsigned int *)(result + 8);
                    if ( v32 < v35 )
                      *(_DWORD *)(result + 8) = v35 - 1;
                    result += 16;
                    if ( (_QWORD *)result == v34 )
                      break;
                    while ( *(_QWORD *)result > 0xFFFFFFFFFFFFFFFDLL )
                    {
                      result += 16;
                      if ( v34 == (_QWORD *)result )
                        return result;
                    }
                  }
                  while ( v34 != (_QWORD *)result );
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v36 = 1;
      while ( v10 != -1 )
      {
        v37 = v36 + 1;
        v8 = v6 & (v36 + v8);
        v9 = (__int64 *)(v4 + 16LL * v8);
        v10 = *v9;
        if ( v7 == *v9 )
          goto LABEL_3;
        v36 = v37;
      }
    }
  }
  return result;
}
