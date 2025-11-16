// Function: sub_D7C840
// Address: 0xd7c840
//
__int64 *__fastcall sub_D7C840(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r12
  int v5; // eax
  int v6; // eax
  int v7; // edx
  __int64 v8; // rdx
  __int64 *v9; // r12
  _BYTE *v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // r15
  __int64 v13; // r14
  size_t v14; // rdi
  __int64 v15; // r13
  __int64 v16; // rax
  void *v17; // r8
  bool v18; // r12
  __m128i *v20; // r14
  unsigned __int64 v21; // rax
  unsigned int v22; // eax
  _QWORD *v23; // rax
  __int64 v24; // r12
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *i; // rdx
  __int64 *v28; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v28 = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= (unsigned int)(3 * v4) )
  {
    v20 = *(__m128i **)(a1 + 8);
    v7 = 2 * v4;
    v21 = (((((((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
              | (unsigned int)(v7 - 1)
              | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
            | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
            | (unsigned int)(v7 - 1)
            | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
            | (unsigned int)(v7 - 1)
            | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
          | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
          | (unsigned int)(v7 - 1)
          | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
            | (unsigned int)(v7 - 1)
            | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
          | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
          | (unsigned int)(v7 - 1)
          | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 8)
        | (((((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
          | (unsigned int)(v7 - 1)
          | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 4)
        | (((unsigned int)(v7 - 1) | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1)) >> 2)
        | (unsigned int)(v7 - 1)
        | ((unsigned __int64)(unsigned int)(v7 - 1) >> 1);
    v22 = v21 + 1;
    if ( v22 < 0x40 )
      v22 = 64;
    *(_DWORD *)(a1 + 24) = v22;
    v23 = (_QWORD *)sub_C7D670(40LL * v22, 8);
    *(_QWORD *)(a1 + 8) = v23;
    if ( !v20 )
    {
      v26 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      for ( i = &v23[5 * v26]; i != v23; v23 += 5 )
      {
        if ( v23 )
        {
          *v23 = 0;
          v23[1] = -1;
          v23[2] = 0;
          v23[3] = 0;
          v23[4] = 0;
        }
      }
      goto LABEL_19;
    }
  }
  else
  {
    v8 = (unsigned int)(v4 - *(_DWORD *)(a1 + 20) - v6);
    if ( (unsigned int)v8 > (unsigned int)v4 >> 3 )
      goto LABEL_3;
    v20 = *(__m128i **)(a1 + 8);
    v25 = ((((((((((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
               | (unsigned int)(v4 - 1)
               | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 4)
             | (((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
             | (unsigned int)(v4 - 1)
             | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 8)
           | (((((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
             | (unsigned int)(v4 - 1)
             | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 4)
           | (((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
           | (unsigned int)(v4 - 1)
           | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 16)
         | (((((((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
             | (unsigned int)(v4 - 1)
             | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 4)
           | (((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
           | (unsigned int)(v4 - 1)
           | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 8)
         | (((((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
           | (unsigned int)(v4 - 1)
           | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 4)
         | (((unsigned int)(v4 - 1) | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1)) >> 2)
         | (unsigned int)(v4 - 1)
         | ((unsigned __int64)(unsigned int)(v4 - 1) >> 1))
        + 1;
    if ( (unsigned int)v25 < 0x40 )
      LODWORD(v25) = 64;
    *(_DWORD *)(a1 + 24) = v25;
    *(_QWORD *)(a1 + 8) = sub_C7D670(40LL * (unsigned int)v25, 8);
    if ( !v20 )
    {
      sub_D7A3B0(a1);
      goto LABEL_19;
    }
  }
  v24 = 40 * v4;
  sub_D7C560(a1, v20, (__m128i *)((char *)v20 + v24));
  sub_C7D6A0((__int64)v20, v24, 8);
LABEL_19:
  sub_D79E80((_DWORD *)a1, a2, &v28);
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  v9 = v28;
  *(_DWORD *)(a1 + 16) = v6;
  v10 = (_BYTE *)v9[3];
  v11 = (_BYTE *)v9[2];
  v12 = *v9;
  v13 = v9[1];
  v14 = v10 - v11;
  if ( v10 == v11 )
  {
    v15 = 0;
    v17 = 0;
  }
  else
  {
    v15 = v9[3] - (_QWORD)v11;
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v14, v11, v8);
    v16 = sub_22077B0(v14);
    v11 = (_BYTE *)v9[2];
    v17 = (void *)v16;
    v10 = (_BYTE *)v9[3];
    v14 = v10 - v11;
  }
  v18 = v13 == -1 && (v14 | v12) == 0;
  if ( v11 == v10 )
  {
    if ( !v18 )
    {
      if ( !v17 )
        goto LABEL_9;
      goto LABEL_8;
    }
    if ( !v17 )
      return v28;
LABEL_25:
    j_j___libc_free_0(v17, v15);
    return v28;
  }
  v17 = memmove(v17, v11, v14);
  if ( v18 )
    goto LABEL_25;
LABEL_8:
  j_j___libc_free_0(v17, v15);
LABEL_9:
  --*(_DWORD *)(a1 + 20);
  return v28;
}
