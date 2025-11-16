// Function: sub_1EE6470
// Address: 0x1ee6470
//
void __fastcall sub_1EE6470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 v7; // r13
  unsigned __int64 v8; // rdx
  unsigned int *v9; // r12
  unsigned int *i; // r15
  unsigned int v11; // eax
  unsigned int v12; // ecx
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int64 v15; // rbx

  v6 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(a1 + 56) )
  {
    *(_QWORD *)(v6 + 192) = sub_1EE6230((_QWORD *)a1);
    v7 = *(_QWORD *)(a1 + 48);
    v8 = *(unsigned int *)(a1 + 104);
    if ( *(_DWORD *)(v7 + 116) >= (unsigned int)v8 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)(v6 + 192) = *(_QWORD *)(a1 + 64);
    v7 = *(_QWORD *)(a1 + 48);
    v8 = *(unsigned int *)(a1 + 104);
    if ( *(_DWORD *)(v7 + 116) >= (unsigned int)v8 )
      goto LABEL_3;
  }
  sub_16CD150(v7 + 104, (const void *)(v7 + 120), v8, 8, a5, a6);
  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(unsigned int *)(a1 + 104);
LABEL_3:
  v9 = *(unsigned int **)(a1 + 96);
  for ( i = &v9[2 * v8]; i != v9; ++*(_DWORD *)(v7 + 112) )
  {
    while ( 1 )
    {
      v11 = *v9;
      v12 = *(_DWORD *)(a1 + 192);
      if ( *v9 >= v12 )
        v11 = (*v9 - v12) | 0x80000000;
      if ( v9[1] )
        break;
      v9 += 2;
      if ( i == v9 )
        return;
    }
    v13 = v11;
    v14 = *(unsigned int *)(v7 + 112);
    v15 = ((unsigned __int64)v9[1] << 32) | v13;
    if ( (unsigned int)v14 >= *(_DWORD *)(v7 + 116) )
    {
      sub_16CD150(v7 + 104, (const void *)(v7 + 120), 0, 8, a5, a6);
      v14 = *(unsigned int *)(v7 + 112);
    }
    v9 += 2;
    *(_QWORD *)(*(_QWORD *)(v7 + 104) + 8 * v14) = v15;
  }
}
