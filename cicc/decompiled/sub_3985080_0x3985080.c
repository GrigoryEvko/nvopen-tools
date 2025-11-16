// Function: sub_3985080
// Address: 0x3985080
//
__int64 __fastcall sub_3985080(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // rcx
  int v6; // eax
  __int64 v7; // r9
  int v8; // eax
  unsigned int v9; // edi
  __int64 *v10; // rcx
  unsigned int v11; // esi
  __int64 v13; // rax
  int v14; // eax
  unsigned int v15; // ecx
  __int64 v16; // rdi
  int v17; // r10d
  __int64 *v18; // r8
  unsigned int v19; // eax
  int v20; // ecx
  int v21; // r10d

  v3 = *a3;
  if ( !a2 )
  {
    if ( !v3 )
      return 0;
    v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 256LL);
    v7 = *(_QWORD *)(v13 + 88);
    v14 = *(_DWORD *)(v13 + 104);
    if ( !v14 )
      return 0;
    v8 = v14 - 1;
LABEL_10:
    LODWORD(v4) = 0;
    v15 = v8 & (((unsigned int)v3 >> 4) ^ ((unsigned int)v3 >> 9));
    v16 = *(_QWORD *)(v7 + 16LL * v15);
    v11 = 0;
    if ( v3 == v16 )
      return (unsigned int)v4;
    goto LABEL_11;
  }
  LODWORD(v4) = 0;
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 256LL);
  v6 = *(_DWORD *)(v5 + 104);
  if ( !v6 )
    return (unsigned int)v4;
  v7 = *(_QWORD *)(v5 + 88);
  v8 = v6 - 1;
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v4 = *v10;
  if ( a2 != *v10 )
  {
    v20 = 1;
    while ( v4 != -8 )
    {
      v21 = v20 + 1;
      v9 = v8 & (v20 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v4 = *v10;
      if ( a2 == *v10 )
        goto LABEL_4;
      v20 = v21;
    }
    LODWORD(v4) = 0;
    if ( !v3 )
      return (unsigned int)v4;
    goto LABEL_10;
  }
LABEL_4:
  v11 = *((_DWORD *)v10 + 2);
  if ( !v3 )
  {
LABEL_5:
    LOBYTE(v4) = v11 != 0;
    return (unsigned int)v4;
  }
  v15 = v8 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v18 = (__int64 *)(v7 + 16LL * v15);
  v16 = *v18;
  if ( v3 != *v18 )
  {
LABEL_11:
    LODWORD(v4) = 1;
    while ( v16 != -8 )
    {
      v17 = v4 + 1;
      v15 = v8 & (v4 + v15);
      v18 = (__int64 *)(v7 + 16LL * v15);
      v16 = *v18;
      if ( v3 == *v18 )
        goto LABEL_14;
      LODWORD(v4) = v17;
    }
    goto LABEL_5;
  }
LABEL_14:
  v19 = *((_DWORD *)v18 + 2);
  if ( !v11 )
    return 0;
  if ( !v19 )
  {
    LODWORD(v4) = 1;
    return (unsigned int)v4;
  }
  LOBYTE(v18) = v19 > v11;
  return (unsigned int)v18;
}
