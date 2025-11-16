// Function: sub_23F9970
// Address: 0x23f9970
//
unsigned __int64 __fastcall sub_23F9970(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int v8; // esi
  __int64 v9; // rcx
  __int64 v10; // r9
  int v11; // r10d
  __int64 *v12; // r15
  int v13; // eax
  __int64 i; // r8
  __int64 *v15; // rdx
  __int64 v16; // rdi
  int v17; // r8d
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 result; // rax
  int v22; // edi
  int v23; // ecx
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 v27; // rcx
  const __m128i *v28; // rdx
  __m128i *v29; // rax
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // rdx
  __int64 *v33; // rdi
  unsigned int j; // eax
  __int64 v35; // rsi
  int v36; // eax
  unsigned __int64 v37; // r14
  __int64 v38; // rdi
  const void *v39; // rsi
  int v40; // edx
  int v41; // edx
  __int64 v42; // rsi
  unsigned int k; // eax
  __int64 v44; // rcx
  int v45; // eax
  int v46; // [rsp+8h] [rbp-58h]
  _QWORD v47[10]; // [rsp+10h] [rbp-50h] BYREF

  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_30;
  }
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v8 - 1;
  v11 = 1;
  v12 = 0;
  v13 = ((0xBF58476D1CE4E5B9LL
        * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
      ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)));
  for ( i = v13 & (v8 - 1); ; i = (unsigned int)v10 & v17 )
  {
    v15 = (__int64 *)(v9 + 24LL * (unsigned int)i);
    v16 = *v15;
    if ( a2 == *v15 && a3 == v15[1] )
    {
      v18 = *((unsigned int *)v15 + 4);
      goto LABEL_12;
    }
    if ( v16 == -4096 )
      break;
    if ( v16 == -8192 && v15[1] == -8192 && !v12 )
      v12 = (__int64 *)(v9 + 24LL * (unsigned int)i);
LABEL_9:
    v17 = v11 + i;
    ++v11;
  }
  if ( v15[1] != -4096 )
    goto LABEL_9;
  v22 = *(_DWORD *)(a1 + 16);
  if ( !v12 )
    v12 = (__int64 *)(v9 + 24LL * (unsigned int)i);
  ++*(_QWORD *)a1;
  v23 = v22 + 1;
  if ( 4 * (v22 + 1) >= 3 * v8 )
  {
LABEL_30:
    sub_23F96A0(a1, 2 * v8);
    v30 = *(_DWORD *)(a1 + 24);
    if ( v30 )
    {
      v31 = v30 - 1;
      i = 1;
      v33 = 0;
      for ( j = v31
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v31 & v36 )
      {
        v32 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v32 + 24LL * j);
        v35 = *v12;
        if ( a2 == *v12 && a3 == v12[1] )
          break;
        if ( v35 == -4096 )
        {
          if ( v12[1] == -4096 )
          {
LABEL_57:
            if ( v33 )
              v12 = v33;
            v23 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_22;
          }
        }
        else if ( v35 == -8192 && v12[1] == -8192 && !v33 )
        {
          v33 = (__int64 *)(v32 + 24LL * j);
        }
        v36 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_53;
    }
LABEL_62:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a1 + 20) - v23 <= v8 >> 3 )
  {
    v46 = v13;
    sub_23F96A0(a1, v8);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v33 = 0;
      i = 1;
      for ( k = v41 & v46; ; k = v41 & v45 )
      {
        v42 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v42 + 24LL * k);
        v44 = *v12;
        if ( a2 == *v12 && a3 == v12[1] )
          break;
        if ( v44 == -4096 )
        {
          if ( v12[1] == -4096 )
            goto LABEL_57;
        }
        else if ( v44 == -8192 && v12[1] == -8192 && !v33 )
        {
          v33 = (__int64 *)(v42 + 24LL * k);
        }
        v45 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_53:
      v23 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_22;
    }
    goto LABEL_62;
  }
LABEL_22:
  *(_DWORD *)(a1 + 16) = v23;
  if ( *v12 != -4096 || v12[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v12 = a2;
  v12[1] = a3;
  *((_DWORD *)v12 + 4) = 0;
  v24 = *(unsigned int *)(a1 + 40);
  v25 = *(unsigned int *)(a1 + 44);
  v47[0] = a2;
  v26 = v24 + 1;
  v47[1] = a3;
  v47[2] = 0;
  if ( v24 + 1 > v25 )
  {
    v37 = *(_QWORD *)(a1 + 32);
    v38 = a1 + 32;
    v39 = (const void *)(a1 + 48);
    if ( v37 > (unsigned __int64)v47 || (unsigned __int64)v47 >= v37 + 24 * v24 )
    {
      sub_C8D5F0(v38, v39, v26, 0x18u, i, v10);
      v27 = *(_QWORD *)(a1 + 32);
      v24 = *(unsigned int *)(a1 + 40);
      v28 = (const __m128i *)v47;
    }
    else
    {
      sub_C8D5F0(v38, v39, v26, 0x18u, i, v10);
      v27 = *(_QWORD *)(a1 + 32);
      v24 = *(unsigned int *)(a1 + 40);
      v28 = (const __m128i *)((char *)v47 + v27 - v37);
    }
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 32);
    v28 = (const __m128i *)v47;
  }
  v29 = (__m128i *)(v27 + 24 * v24);
  *v29 = _mm_loadu_si128(v28);
  v29[1].m128i_i64[0] = v28[1].m128i_i64[0];
  v18 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v18 + 1;
  *((_DWORD *)v12 + 4) = v18;
LABEL_12:
  v19 = *(_QWORD *)(a1 + 32) + 24 * v18;
  v20 = *(_QWORD *)(v19 + 16);
  result = v20 + a4;
  if ( v20 < a4 )
    v20 = a4;
  if ( result < v20 )
    result = -1;
  *(_QWORD *)(v19 + 16) = result;
  return result;
}
