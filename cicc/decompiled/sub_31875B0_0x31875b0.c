// Function: sub_31875B0
// Address: 0x31875b0
//
__int64 __fastcall sub_31875B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rsi
  int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rdx
  char *v11; // r14
  char *v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r10d
  __int64 *v15; // rcx
  __int64 v16; // r11
  __int64 v17; // rcx
  _QWORD *v18; // rbx
  __int64 v19; // r8
  unsigned __int64 v20; // r12
  __m128i v21; // xmm0
  __int64 v22; // rcx
  __m128i v23; // xmm2
  __int64 v24; // rdx
  __m128i v25; // xmm0
  void (__fastcall *v26)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  void (__fastcall *v30)(__int64, __int64, __int64); // rcx
  unsigned __int64 v31; // r14
  _QWORD *v32; // rcx
  _QWORD *v33; // rdx
  unsigned __int64 v34; // rsi
  int v35; // ecx
  int v36; // r10d
  int v37; // ecx
  int v38; // ebx
  __m128i v39; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v40)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp-48h] [rbp-48h]
  __int64 v41; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a1 + 352);
  v4 = *(_QWORD *)(a1 + 336);
  if ( (_DWORD)result )
  {
    v6 = result - 1;
    v7 = (result - 1) & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      result = v4 + 16 * result;
      if ( v8 != (__int64 *)result )
      {
        v10 = *(_QWORD *)(a1 + 360);
        v11 = (char *)(v10 + 40LL * *((unsigned int *)v8 + 2));
        result = *(unsigned int *)(a1 + 368);
        v12 = (char *)(v10 + 40 * result);
        if ( v12 != v11 )
        {
          v13 = *(_QWORD *)v11;
          v14 = v6 & (((0xBF58476D1CE4E5B9LL * *(_QWORD *)v11) >> 31) ^ (484763065 * *(_DWORD *)v11));
          v15 = (__int64 *)(v4 + 16LL * v14);
          v16 = *v15;
          if ( *v15 == *(_QWORD *)v11 )
          {
LABEL_6:
            *v15 = -2;
            v17 = *(unsigned int *)(a1 + 368);
            --*(_DWORD *)(a1 + 344);
            v10 = *(_QWORD *)(a1 + 360);
            ++*(_DWORD *)(a1 + 348);
            LODWORD(result) = v17;
            v12 = (char *)(v10 + 40 * v17);
          }
          else
          {
            v37 = 1;
            while ( v16 != -1 )
            {
              v38 = v37 + 1;
              v14 = v6 & (v37 + v14);
              v15 = (__int64 *)(v4 + 16LL * v14);
              v16 = *v15;
              if ( v13 == *v15 )
                goto LABEL_6;
              v37 = v38;
            }
          }
          v18 = v11 + 40;
          v19 = v12 - (v11 + 40);
          v20 = 0xCCCCCCCCCCCCCCCDLL * (v19 >> 3);
          if ( v19 > 0 )
          {
            do
            {
              v21 = _mm_loadu_si128((const __m128i *)(v18 + 1));
              *(v18 - 5) = *v18;
              *(__m128i *)(v18 + 1) = _mm_loadu_si128(&v39);
              v39 = v21;
              v22 = v18[3];
              v23 = _mm_loadu_si128((const __m128i *)v18 - 2);
              v18[3] = 0;
              v24 = v18[4];
              v18[4] = v41;
              v25 = _mm_loadu_si128(&v39);
              v39 = v23;
              v26 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))*(v18 - 2);
              *((__m128i *)v18 - 2) = v25;
              v40 = v26;
              *(v18 - 2) = v22;
              v27 = *(v18 - 1);
              v41 = v27;
              *(v18 - 1) = v24;
              if ( v26 )
                v26(&v39, &v39, 3, v27, v19, v13, v39.m128i_i64[0], v39.m128i_i64[1], v40);
              v18 += 5;
              --v20;
            }
            while ( v20 );
            LODWORD(result) = *(_DWORD *)(a1 + 368);
            v10 = *(_QWORD *)(a1 + 360);
          }
          v28 = (unsigned int)(result - 1);
          *(_DWORD *)(a1 + 368) = v28;
          v29 = v10 + 40 * v28;
          v30 = *(void (__fastcall **)(__int64, __int64, __int64))(v29 + 24);
          if ( v30 )
          {
            v30(v29 + 8, v29 + 8, 3);
            v10 = *(_QWORD *)(a1 + 360);
          }
          result = v10 + 40LL * *(unsigned int *)(a1 + 368);
          if ( v11 != (char *)result )
          {
            v31 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v11[-v10] >> 3);
            result = *(unsigned int *)(a1 + 344);
            if ( (_DWORD)result )
            {
              v32 = *(_QWORD **)(a1 + 336);
              v33 = &v32[2 * *(unsigned int *)(a1 + 352)];
              if ( v32 != v33 )
              {
                while ( 1 )
                {
                  result = (__int64)v32;
                  if ( *v32 <= 0xFFFFFFFFFFFFFFFDLL )
                    break;
                  v32 += 2;
                  if ( v33 == v32 )
                    return result;
                }
                if ( v32 != v33 )
                {
                  do
                  {
                    v34 = *(unsigned int *)(result + 8);
                    if ( v31 < v34 )
                      *(_DWORD *)(result + 8) = v34 - 1;
                    result += 16;
                    if ( (_QWORD *)result == v33 )
                      break;
                    while ( *(_QWORD *)result > 0xFFFFFFFFFFFFFFFDLL )
                    {
                      result += 16;
                      if ( v33 == (_QWORD *)result )
                        return result;
                    }
                  }
                  while ( v33 != (_QWORD *)result );
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v35 = 1;
      while ( v9 != -1 )
      {
        v36 = v35 + 1;
        v7 = v6 & (v35 + v7);
        v8 = (__int64 *)(v4 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v35 = v36;
      }
    }
  }
  return result;
}
