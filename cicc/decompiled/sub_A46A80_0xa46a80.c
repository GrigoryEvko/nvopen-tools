// Function: sub_A46A80
// Address: 0xa46a80
//
__int64 __fastcall sub_A46A80(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  int v8; // r10d
  __int64 v9; // r8
  _QWORD *v10; // r13
  unsigned int v11; // r14d
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r13
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rsi
  int v24; // r9d
  _QWORD *v25; // r8
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  _QWORD *v30; // rdi
  unsigned int v31; // r14d
  __int64 v32; // rcx
  _QWORD v33[7]; // [rsp+8h] [rbp-38h] BYREF

  v5 = a1 + 256;
  v7 = *(_DWORD *)(a1 + 280);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 256);
    goto LABEL_20;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 264);
  v10 = 0;
  v11 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v12 = (v7 - 1) & v11;
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( a3 != *v13 )
  {
    while ( v14 != -4096 )
    {
      if ( !v10 && v14 == -8192 )
        v10 = v13;
      v12 = (v7 - 1) & (v8 + v12);
      v13 = (__int64 *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( a3 == *v13 )
        goto LABEL_3;
      ++v8;
    }
    if ( !v10 )
      v10 = v13;
    v17 = *(_DWORD *)(a1 + 272);
    ++*(_QWORD *)(a1 + 256);
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 276) - v18 > v7 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 272) = v18;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 276);
        *v10 = a3;
        v15 = v10 + 1;
        *v15 = 0;
        goto LABEL_18;
      }
      sub_A42F50(v5, v7);
      v26 = *(_DWORD *)(a1 + 280);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a1 + 264);
        v29 = 1;
        v30 = 0;
        v31 = v27 & v11;
        v18 = *(_DWORD *)(a1 + 272) + 1;
        v10 = (_QWORD *)(v28 + 16LL * v31);
        v32 = *v10;
        if ( a3 != *v10 )
        {
          while ( v32 != -4096 )
          {
            if ( !v30 && v32 == -8192 )
              v30 = v10;
            v31 = v27 & (v29 + v31);
            v10 = (_QWORD *)(v28 + 16LL * v31);
            v32 = *v10;
            if ( a3 == *v10 )
              goto LABEL_15;
            ++v29;
          }
          if ( v30 )
            v10 = v30;
        }
        goto LABEL_15;
      }
LABEL_43:
      ++*(_DWORD *)(a1 + 272);
      BUG();
    }
LABEL_20:
    sub_A42F50(v5, 2 * v7);
    v19 = *(_DWORD *)(a1 + 280);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 264);
      v22 = (v19 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = *(_DWORD *)(a1 + 272) + 1;
      v10 = (_QWORD *)(v21 + 16LL * v22);
      v23 = *v10;
      if ( a3 != *v10 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( !v25 && v23 == -8192 )
            v25 = v10;
          v22 = v20 & (v24 + v22);
          v10 = (_QWORD *)(v21 + 16LL * v22);
          v23 = *v10;
          if ( a3 == *v10 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v10 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_43;
  }
LABEL_3:
  v15 = v13 + 1;
  result = *((unsigned int *)v13 + 3);
  if ( !(_DWORD)result )
  {
LABEL_18:
    v33[0] = a3;
    sub_A3DCA0(a1 + 208, v33);
    *(_DWORD *)v15 = a2;
    *((_DWORD *)v15 + 1) = (__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 3;
    return sub_A45280(a1, *(unsigned __int8 **)(a3 + 136));
  }
  return result;
}
