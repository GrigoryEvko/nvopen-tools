// Function: sub_1E70230
// Address: 0x1e70230
//
void __fastcall sub_1E70230(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned int v9; // esi
  __int64 *v10; // rax
  __int64 v11; // r9
  unsigned __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  unsigned __int64 i; // rdx
  __int64 v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // r9
  __int64 v21; // r12
  __int64 j; // r14
  int v23; // eax
  int v24; // r10d
  int v25; // eax
  int v26; // r10d

  v4 = sub_1E6BEE0(a2[116], a2[117]);
  if ( a2[117] == v4 )
    return;
  v5 = v4;
  v6 = *(_QWORD *)(a2[264] + 272LL);
  if ( (*(_BYTE *)(v4 + 46) & 4) != 0 )
  {
    do
      v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v5 + 46) & 4) != 0 );
  }
  v7 = *(_QWORD *)(v6 + 368);
  v8 = *(unsigned int *)(v6 + 384);
  if ( (_DWORD)v8 )
  {
    v9 = (v8 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
      goto LABEL_6;
    v23 = 1;
    while ( v11 != -8 )
    {
      v24 = v23 + 1;
      v9 = (v8 - 1) & (v23 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == v5 )
        goto LABEL_6;
      v23 = v24;
    }
  }
  v10 = (__int64 *)(v7 + 16 * v8);
LABEL_6:
  *(_QWORD *)(a1 + 8) = v10[1];
  v12 = sub_1E6C1C0(a2[117], a2[116]);
  v14 = *(_QWORD *)(v13 + 272);
  for ( i = v12; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v16 = *(_QWORD *)(v14 + 368);
  v17 = *(unsigned int *)(v14 + 384);
  if ( (_DWORD)v17 )
  {
    v18 = (v17 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == i )
      goto LABEL_10;
    v25 = 1;
    while ( v20 != -8 )
    {
      v26 = v25 + 1;
      v18 = (v17 - 1) & (v25 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( *v19 == i )
        goto LABEL_10;
      v25 = v26;
    }
  }
  v19 = (__int64 *)(v16 + 16 * v17);
LABEL_10:
  *(_QWORD *)(a1 + 16) = v19[1];
  v21 = a2[6];
  for ( j = a2[7]; j != v21; v21 += 272 )
  {
    if ( **(_WORD **)(*(_QWORD *)(v21 + 8) + 16LL) == 15 )
      sub_1E6F590(a1, v21, (__int64)a2);
  }
}
