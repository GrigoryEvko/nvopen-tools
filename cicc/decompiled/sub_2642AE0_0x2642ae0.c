// Function: sub_2642AE0
// Address: 0x2642ae0
//
__int64 __fastcall sub_2642AE0(__int64 a1, __int64 a2, __int64 a3)
{
  int *v3; // rax
  int *v5; // rdx
  unsigned int v8; // r8d
  int v9; // r12d
  __int64 v10; // r15
  int v11; // r14d
  int v12; // ecx
  unsigned int v13; // r9d
  int v14; // edi
  __int64 v15; // rdi
  __int64 v16; // r11
  unsigned int v17; // esi
  int *v18; // r9
  int v19; // ebx
  int v20; // r9d
  int v21; // r11d
  __int64 v22; // [rsp-8h] [rbp-8h]

  v3 = *(int **)(a2 + 8);
  v5 = &v3[*(unsigned int *)(a2 + 24)];
  if ( !*(_DWORD *)(a2 + 16) || v3 == v5 )
    return 0;
  while ( (unsigned int)*v3 > 0xFFFFFFFD )
  {
    if ( ++v3 == v5 )
      return 0;
  }
  if ( v3 == v5 )
    return 0;
  v8 = 0;
  v9 = *(_DWORD *)(a3 + 24);
  v10 = *(_QWORD *)(a3 + 8);
  v11 = v9 - 1;
  do
  {
    v12 = *v3;
    if ( !v9 )
      goto LABEL_14;
    v13 = v11 & (37 * v12);
    v14 = *(_DWORD *)(v10 + 4LL * v13);
    if ( v12 != v14 )
    {
      v21 = 1;
      while ( v14 != -1 )
      {
        v13 = v11 & (v21 + v13);
        v14 = *(_DWORD *)(v10 + 4LL * v13);
        if ( v12 == v14 )
          goto LABEL_11;
        ++v21;
      }
      goto LABEL_14;
    }
LABEL_11:
    v15 = *(unsigned int *)(a1 + 152);
    v16 = *(_QWORD *)(a1 + 136);
    if ( (_DWORD)v15 )
    {
      v17 = (v15 - 1) & (37 * v12);
      v18 = (int *)(v16 + 8LL * v17);
      v19 = *v18;
      if ( v12 == *v18 )
        goto LABEL_13;
      v20 = 1;
      while ( v19 != -1 )
      {
        v17 = (v15 - 1) & (v20 + v17);
        *((_DWORD *)&v22 - 11) = v20 + 1;
        v18 = (int *)(v16 + 8LL * v17);
        v19 = *v18;
        if ( v12 == *v18 )
          goto LABEL_13;
        v20 = *((_DWORD *)&v22 - 11);
      }
    }
    v18 = (int *)(v16 + 8 * v15);
LABEL_13:
    LOBYTE(v8) = *((_BYTE *)v18 + 4) | v8;
    if ( (_BYTE)v8 == 3 )
      return v8;
LABEL_14:
    if ( ++v3 == v5 )
      break;
    while ( (unsigned int)*v3 > 0xFFFFFFFD )
    {
      if ( v5 == ++v3 )
        return v8;
    }
  }
  while ( v5 != v3 );
  return v8;
}
