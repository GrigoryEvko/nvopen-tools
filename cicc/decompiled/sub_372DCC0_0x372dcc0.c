// Function: sub_372DCC0
// Address: 0x372dcc0
//
__int64 __fastcall sub_372DCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 result; // rax
  int v20; // ecx
  int v21; // ecx
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  const __m128i *v26; // rdx
  __m128i *v27; // rax
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // rdx
  __int64 *v31; // rdi
  unsigned int j; // eax
  __int64 v33; // rsi
  int v34; // eax
  unsigned __int64 v35; // r14
  __int64 v36; // rdi
  const void *v37; // rsi
  int v38; // edx
  int v39; // edx
  __int64 v40; // rsi
  unsigned int k; // eax
  __int64 v42; // rcx
  int v43; // eax
  int v44; // [rsp+8h] [rbp-58h]
  _QWORD v45[10]; // [rsp+10h] [rbp-50h] BYREF

  v8 = *(_DWORD *)(a1 + 24);
  if ( !v8 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_26;
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
    if ( *v15 == a2 && v15[1] == a3 )
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
  v20 = *(_DWORD *)(a1 + 16);
  if ( !v12 )
    v12 = v15;
  ++*(_QWORD *)a1;
  v21 = v20 + 1;
  if ( 4 * v21 >= 3 * v8 )
  {
LABEL_26:
    sub_372C8B0(a1, 2 * v8);
    v28 = *(_DWORD *)(a1 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      i = 1;
      v31 = 0;
      for ( j = v29
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v29 & v34 )
      {
        v30 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v30 + 24LL * j);
        v33 = *v12;
        if ( *v12 == a2 && a3 == v12[1] )
          break;
        if ( v33 == -4096 )
        {
          if ( v12[1] == -4096 )
          {
LABEL_53:
            if ( v31 )
              v12 = v31;
            v21 = *(_DWORD *)(a1 + 16) + 1;
            goto LABEL_18;
          }
        }
        else if ( v33 == -8192 && v12[1] == -8192 && !v31 )
        {
          v31 = (__int64 *)(v30 + 24LL * j);
        }
        v34 = i + j;
        i = (unsigned int)(i + 1);
      }
      goto LABEL_49;
    }
LABEL_58:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v8 - *(_DWORD *)(a1 + 20) - v21 <= v8 >> 3 )
  {
    v44 = v13;
    sub_372C8B0(a1, v8);
    v38 = *(_DWORD *)(a1 + 24);
    if ( v38 )
    {
      v39 = v38 - 1;
      v31 = 0;
      i = 1;
      for ( k = v39 & v44; ; k = v39 & v43 )
      {
        v40 = *(_QWORD *)(a1 + 8);
        v12 = (__int64 *)(v40 + 24LL * k);
        v42 = *v12;
        if ( *v12 == a2 && a3 == v12[1] )
          break;
        if ( v42 == -4096 )
        {
          if ( v12[1] == -4096 )
            goto LABEL_53;
        }
        else if ( v42 == -8192 && v12[1] == -8192 && !v31 )
        {
          v31 = (__int64 *)(v40 + 24LL * k);
        }
        v43 = i + k;
        i = (unsigned int)(i + 1);
      }
LABEL_49:
      v21 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_18;
    }
    goto LABEL_58;
  }
LABEL_18:
  *(_DWORD *)(a1 + 16) = v21;
  if ( *v12 != -4096 || v12[1] != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v12 = a2;
  v12[1] = a3;
  *((_DWORD *)v12 + 4) = 0;
  v22 = *(unsigned int *)(a1 + 40);
  v23 = *(unsigned int *)(a1 + 44);
  v45[0] = a2;
  v24 = v22 + 1;
  v45[1] = a3;
  v45[2] = 0;
  if ( v22 + 1 > v23 )
  {
    v35 = *(_QWORD *)(a1 + 32);
    v36 = a1 + 32;
    v37 = (const void *)(a1 + 48);
    if ( v35 > (unsigned __int64)v45 || (unsigned __int64)v45 >= v35 + 24 * v22 )
    {
      sub_C8D5F0(v36, v37, v24, 0x18u, i, v10);
      v25 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v26 = (const __m128i *)v45;
    }
    else
    {
      sub_C8D5F0(v36, v37, v24, 0x18u, i, v10);
      v25 = *(_QWORD *)(a1 + 32);
      v22 = *(unsigned int *)(a1 + 40);
      v26 = (const __m128i *)((char *)v45 + v25 - v35);
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = (const __m128i *)v45;
  }
  v27 = (__m128i *)(v25 + 24 * v22);
  *v27 = _mm_loadu_si128(v26);
  v27[1].m128i_i64[0] = v26[1].m128i_i64[0];
  v18 = *(unsigned int *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v18 + 1;
  *((_DWORD *)v12 + 4) = v18;
LABEL_12:
  result = *(_QWORD *)(a1 + 32) + 24 * v18;
  *(_QWORD *)(result + 16) = a4;
  return result;
}
