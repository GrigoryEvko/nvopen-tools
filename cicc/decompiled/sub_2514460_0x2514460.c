// Function: sub_2514460
// Address: 0x2514460
//
__int64 __fastcall sub_2514460(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  unsigned int v8; // esi
  __int64 v9; // rdx
  int v10; // r10d
  __int64 *v11; // r9
  unsigned int v12; // eax
  __int64 *v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rdi
  unsigned __int64 *v16; // rbx
  unsigned int v17; // r8d
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // r13
  int v22; // eax
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // [rsp-38h] [rbp-38h] BYREF
  __int64 *v26; // [rsp-30h] [rbp-30h] BYREF

  if ( !(_DWORD)qword_4FEF0E8 )
    return 0;
  v7 = *a1;
  v25 = a4;
  v8 = *(_DWORD *)(v7 + 24);
  if ( !v8 )
  {
    v26 = 0;
    ++*(_QWORD *)v7;
LABEL_35:
    v8 *= 2;
    goto LABEL_36;
  }
  v9 = *(_QWORD *)(v7 + 8);
  v10 = 1;
  v11 = 0;
  v12 = (v8 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( a4 == *v13 )
    goto LABEL_5;
  while ( v14 != -4096 )
  {
    if ( v14 == -8192 && !v11 )
      v11 = v13;
    v12 = (v8 - 1) & (v10 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( a4 == *v13 )
      goto LABEL_5;
    ++v10;
  }
  v22 = *(_DWORD *)(v7 + 16);
  if ( !v11 )
    v11 = v13;
  v23 = v22 + 1;
  v26 = v11;
  ++*(_QWORD *)v7;
  if ( 4 * (v22 + 1) >= 3 * v8 )
    goto LABEL_35;
  if ( v8 - *(_DWORD *)(v7 + 20) - v23 <= v8 >> 3 )
  {
LABEL_36:
    sub_2514230(v7, v8);
    sub_2510A70(v7, &v25, &v26);
    v23 = *(_DWORD *)(v7 + 16) + 1;
  }
  v13 = v26;
  *(_DWORD *)(v7 + 16) = v23;
  if ( *v13 != -4096 )
    --*(_DWORD *)(v7 + 20);
  v24 = v25;
  v13[1] = 0;
  *v13 = v24;
LABEL_5:
  v15 = v13[1];
  v16 = (unsigned __int64 *)(v13 + 1);
  if ( !v15 )
  {
    v20 = sub_22077B0(0x60u);
    v15 = v20;
    if ( v20 )
    {
      *(_QWORD *)v20 = 0;
      *(_QWORD *)(v20 + 8) = v20 + 32;
      *(_QWORD *)(v20 + 16) = 8;
      *(_DWORD *)(v20 + 24) = 0;
      *(_BYTE *)(v20 + 28) = 1;
    }
    v21 = *v16;
    *v16 = v20;
    if ( v21 )
    {
      if ( !*(_BYTE *)(v21 + 28) )
        _libc_free(*(_QWORD *)(v21 + 8));
      j_j___libc_free_0(v21);
      v15 = *v16;
    }
  }
  if ( *(_DWORD *)(v15 + 20) - *(_DWORD *)(v15 + 24) < (unsigned int)qword_4FEF0E8 )
  {
    sub_AE6EC0(v15, a5);
    return 1;
  }
  else
  {
    v17 = *(unsigned __int8 *)(v15 + 28);
    if ( (_BYTE)v17 )
    {
      v18 = *(_QWORD **)(v15 + 8);
      v19 = &v18[*(unsigned int *)(v15 + 20)];
      if ( v18 == v19 )
        return 0;
      while ( a5 != *v18 )
      {
        if ( v19 == ++v18 )
          return 0;
      }
    }
    else
    {
      LOBYTE(v17) = sub_C8CA60(v15, a5) != 0;
    }
  }
  return v17;
}
