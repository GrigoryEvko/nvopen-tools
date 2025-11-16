// Function: sub_2577D20
// Address: 0x2577d20
//
char __fastcall sub_2577D20(__int64 a1, __int64 a2)
{
  __m128i *v2; // rax
  __int64 v3; // r15
  __m128i *v4; // r12
  __m128i *v5; // rbx
  __int64 v6; // r13
  unsigned int v7; // eax
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *i; // rdx
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // r14
  const __m128i *v16; // r12
  const __m128i *v17; // rbx
  __int64 v18; // r13
  __int64 *v19; // rsi
  __int64 v20; // rax
  __m128i *v21; // r9
  int v22; // r13d
  int v23; // eax
  size_t v24; // rdx
  char *v25; // rdi
  int v26; // r10d
  unsigned int k; // r8d
  __m128i *v28; // r14
  const void *v29; // rcx
  unsigned int v30; // r8d
  int v31; // eax
  __int64 v32; // rdx
  _QWORD *j; // rdx
  size_t v35; // [rsp-80h] [rbp-80h]
  const void *v36; // [rsp-78h] [rbp-78h]
  int v37; // [rsp-60h] [rbp-60h]
  unsigned int v38; // [rsp-5Ch] [rbp-5Ch]
  __m128i *v39; // [rsp-58h] [rbp-58h]
  __m128i *v40; // [rsp-58h] [rbp-58h]
  __int64 v41; // [rsp-50h] [rbp-50h]
  __m128i *v42; // [rsp-50h] [rbp-50h]
  __int64 v43; // [rsp-50h] [rbp-50h]
  __m128i *v44; // [rsp-48h] [rbp-48h] BYREF
  __m128i *v45; // [rsp-40h] [rbp-40h] BYREF

  LODWORD(v2) = *(_DWORD *)(a2 + 16);
  if ( (_DWORD)v2 )
  {
    v3 = a1;
    v2 = *(__m128i **)(a2 + 8);
    v4 = &v2[*(unsigned int *)(a2 + 24)];
    if ( v2 != v4 )
    {
      while ( 1 )
      {
        v5 = v2;
        if ( v2->m128i_i64[0] < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++v2 )
          return (char)v2;
      }
      if ( v4 != v2 )
      {
        v6 = *(unsigned int *)(a1 + 24);
        if ( !(_DWORD)v6 )
        {
          while ( 1 )
          {
            ++*(_QWORD *)v3;
            v44 = 0;
LABEL_9:
            v41 = *(_QWORD *)(v3 + 8);
            v7 = sub_AF1560((unsigned int)(2 * v6 - 1));
            if ( v7 < 0x40 )
              v7 = 64;
            *(_DWORD *)(v3 + 24) = v7;
            v8 = (_QWORD *)sub_C7D670(16LL * v7, 8);
            v9 = v41;
            *(_QWORD *)(v3 + 8) = v8;
            if ( v41 )
            {
              v10 = *(unsigned int *)(v3 + 24);
              v11 = 16 * v6;
              *(_QWORD *)(v3 + 16) = 0;
              for ( i = &v8[2 * v10]; i != v8; v8 += 2 )
              {
                if ( v8 )
                {
                  *v8 = -1;
                  v8[1] = 0;
                }
              }
              if ( v41 != v41 + v11 )
              {
                v13 = v3;
                v42 = v4;
                v14 = v11;
                v15 = v9;
                v39 = v5;
                v16 = (const __m128i *)(v9 + v11);
                v17 = (const __m128i *)v9;
                v18 = v13;
                do
                {
                  while ( v17->m128i_i64[0] == -1 || v17->m128i_i64[0] == -2 )
                  {
                    if ( v16 == ++v17 )
                      goto LABEL_22;
                  }
                  v19 = (__int64 *)v17++;
                  sub_B9B010(v18, v19, &v45);
                  *v45 = _mm_loadu_si128(v17 - 1);
                  ++*(_DWORD *)(v18 + 16);
                }
                while ( v16 != v17 );
LABEL_22:
                v20 = v18;
                v4 = v42;
                v5 = v39;
                v11 = v14;
                v9 = v15;
                v3 = v20;
              }
              sub_C7D6A0(v9, v11, 8);
              goto LABEL_24;
            }
            v32 = *(unsigned int *)(v3 + 24);
            *(_QWORD *)(v3 + 16) = 0;
            for ( j = &v8[2 * v32]; j != v8; v8 += 2 )
            {
              if ( v8 )
              {
                *v8 = -1;
                v8[1] = 0;
              }
            }
LABEL_24:
            sub_B9B010(v3, v5, &v44);
            v21 = v44;
            LODWORD(v2) = *(_DWORD *)(v3 + 16) + 1;
LABEL_25:
            *(_DWORD *)(v3 + 16) = (_DWORD)v2;
            if ( v21->m128i_i64[0] != -1 )
              --*(_DWORD *)(v3 + 20);
            *v21 = _mm_loadu_si128(v5);
LABEL_28:
            if ( ++v5 == v4 )
              break;
            while ( 1 )
            {
              v2 = (__m128i *)v5->m128i_i64[0];
              if ( v5->m128i_i64[0] != -1 && v2 != (__m128i *)-2LL )
                break;
              if ( v4 == ++v5 )
                return (char)v2;
            }
            if ( v4 == v5 )
              break;
            v6 = *(unsigned int *)(v3 + 24);
            if ( (_DWORD)v6 )
              goto LABEL_35;
          }
          return (char)v2;
        }
LABEL_35:
        v22 = v6 - 1;
        v43 = *(_QWORD *)(v3 + 8);
        v23 = sub_C94890(v5->m128i_i64[0], v5->m128i_i64[1]);
        v24 = v5->m128i_u64[1];
        v25 = (char *)v5->m128i_i64[0];
        v21 = 0;
        v26 = 1;
        for ( k = v22 & v23; ; k = v22 & v30 )
        {
          v28 = (__m128i *)(v43 + 16LL * k);
          v29 = (const void *)v28->m128i_i64[0];
          LOBYTE(v2) = v25 + 1 == 0;
          if ( v28->m128i_i64[0] != -1 )
          {
            LOBYTE(v2) = v25 + 2 == 0;
            if ( v29 != (const void *)-2LL )
            {
              if ( v28->m128i_i64[1] != v24 )
                goto LABEL_39;
              v37 = v26;
              v38 = k;
              v40 = v21;
              if ( !v24 )
                goto LABEL_28;
              v35 = v24;
              v36 = (const void *)v28->m128i_i64[0];
              LODWORD(v2) = memcmp(v25, v29, v24);
              v29 = v36;
              v24 = v35;
              v21 = v40;
              k = v38;
              v26 = v37;
              LOBYTE(v2) = (_DWORD)v2 == 0;
            }
          }
          if ( (_BYTE)v2 )
            goto LABEL_28;
          if ( v29 == (const void *)-1LL )
          {
            v31 = *(_DWORD *)(v3 + 16);
            v6 = *(unsigned int *)(v3 + 24);
            if ( !v21 )
              v21 = v28;
            ++*(_QWORD *)v3;
            LODWORD(v2) = v31 + 1;
            v44 = v21;
            if ( 4 * (int)v2 < (unsigned int)(3 * v6) )
            {
              if ( (int)v6 - ((int)v2 + *(_DWORD *)(v3 + 20)) <= (unsigned int)v6 >> 3 )
              {
                sub_BA8070(v3, v6);
                goto LABEL_24;
              }
              goto LABEL_25;
            }
            goto LABEL_9;
          }
LABEL_39:
          if ( v29 != (const void *)-2LL || v21 )
            v28 = v21;
          v30 = v26 + k;
          v21 = v28;
          ++v26;
        }
      }
    }
  }
  return (char)v2;
}
