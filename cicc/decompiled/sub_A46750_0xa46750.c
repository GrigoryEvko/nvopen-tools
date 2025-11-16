// Function: sub_A46750
// Address: 0xa46750
//
__int64 __fastcall sub_A46750(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r10d
  _QWORD *v10; // r13
  unsigned int v11; // ebx
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned int *v15; // rdi
  __int64 result; // rax
  int v17; // eax
  int v18; // edx
  __int64 *v19; // rbx
  __int64 *i; // r13
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // r9d
  _QWORD *v27; // r8
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  int v31; // r8d
  unsigned int v32; // ebx
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  unsigned int *v35; // [rsp+8h] [rbp-48h]
  _QWORD v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1 + 256;
  v7 = *(_DWORD *)(a1 + 280);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 256);
    goto LABEL_24;
  }
  v8 = *(_QWORD *)(a1 + 264);
  v9 = 1;
  v10 = 0;
  v11 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v12 = (v7 - 1) & v11;
  v13 = (__int64 *)(v8 + 16LL * v12);
  v14 = *v13;
  if ( a3 == *v13 )
  {
LABEL_3:
    v15 = (unsigned int *)(v13 + 1);
    result = *((unsigned int *)v13 + 3);
    v35 = v15;
    if ( (_DWORD)result )
      return result;
    goto LABEL_18;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = (v7 - 1) & (v9 + v12);
    v13 = (__int64 *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( a3 == *v13 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v13;
  v17 = *(_DWORD *)(a1 + 272);
  ++*(_QWORD *)(a1 + 256);
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v7 )
  {
LABEL_24:
    sub_A42F50(v5, 2 * v7);
    v21 = *(_DWORD *)(a1 + 280);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 264);
      v24 = (v21 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = *(_DWORD *)(a1 + 272) + 1;
      v10 = (_QWORD *)(v23 + 16LL * v24);
      v25 = *v10;
      if ( a3 != *v10 )
      {
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( !v27 && v25 == -8192 )
            v27 = v10;
          v24 = v22 & (v26 + v24);
          v10 = (_QWORD *)(v23 + 16LL * v24);
          v25 = *v10;
          if ( a3 == *v10 )
            goto LABEL_15;
          ++v26;
        }
        if ( v27 )
          v10 = v27;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  if ( v7 - *(_DWORD *)(a1 + 276) - v18 <= v7 >> 3 )
  {
    sub_A42F50(v5, v7);
    v28 = *(_DWORD *)(a1 + 280);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 264);
      v31 = 1;
      v32 = v29 & v11;
      v18 = *(_DWORD *)(a1 + 272) + 1;
      v33 = 0;
      v10 = (_QWORD *)(v30 + 16LL * v32);
      v34 = *v10;
      if ( a3 != *v10 )
      {
        while ( v34 != -4096 )
        {
          if ( !v33 && v34 == -8192 )
            v33 = v10;
          v32 = v29 & (v31 + v32);
          v10 = (_QWORD *)(v30 + 16LL * v32);
          v34 = *v10;
          if ( a3 == *v10 )
            goto LABEL_15;
          ++v31;
        }
        if ( v33 )
          v10 = v33;
      }
      goto LABEL_15;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 272);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 272) = v18;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 276);
  *v10 = a3;
  v10[1] = 0;
  v35 = (unsigned int *)(v10 + 1);
LABEL_18:
  v19 = *(__int64 **)(a3 + 136);
  for ( i = &v19[*(unsigned int *)(a3 + 144)]; i != v19; ++v19 )
  {
    if ( *(_BYTE *)*v19 != 2 )
      sub_A45E40(a1, a2, *v19);
  }
  v36[0] = a3;
  sub_A3DCA0(a1 + 208, v36);
  *v35 = a2;
  result = (__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 3;
  v35[1] = result;
  return result;
}
