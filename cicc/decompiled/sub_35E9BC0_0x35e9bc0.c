// Function: sub_35E9BC0
// Address: 0x35e9bc0
//
__int64 __fastcall sub_35E9BC0(__int64 a1, unsigned int a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // ecx
  __int64 v7; // r13
  unsigned int v8; // r12d
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rdi
  int v12; // esi
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r9
  int v16; // eax
  int v17; // r14d
  __int64 v18; // rbx
  __int64 i; // rdx
  __int64 *v20; // rax
  int v21; // eax
  int v22; // eax
  int v24; // eax
  int v25; // r10d
  __int64 v26; // [rsp+8h] [rbp-48h]
  int v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  int v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  int v31; // [rsp+18h] [rbp-38h]

  v29 = a3 + 1;
  v4 = sub_35E71C0(a1, a2, *(_DWORD *)(a1 + 6436));
  v26 = v5;
  if ( v4 == v5 )
    return 0;
  v6 = v29;
  v7 = v4;
  v8 = 0;
  do
  {
    v9 = *(_QWORD *)(a1 + 72);
    v10 = *(_DWORD *)(v9 + 960);
    v11 = *(_QWORD *)(v9 + 944);
    if ( !v10 )
      goto LABEL_29;
    v12 = v10 - 1;
    v13 = v12 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v14 = (__int64 *)(v11 + 16LL * v13);
    v15 = *v14;
    if ( *v14 != v7 )
    {
      v24 = 1;
      while ( v15 != -4096 )
      {
        v25 = v24 + 1;
        v13 = v12 & (v24 + v13);
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( *v14 == v7 )
          goto LABEL_5;
        v24 = v25;
      }
LABEL_29:
      sub_35E8960(a1, v7);
      BUG();
    }
LABEL_5:
    v27 = v6;
    v30 = v14[1];
    v16 = sub_35E8960(a1, v7);
    v6 = v27;
    v17 = v16;
    v18 = *(_QWORD *)(v30 + 120);
    for ( i = v18 + 16LL * *(unsigned int *)(v30 + 128); i != v18; v18 += 16 )
    {
      while ( (((unsigned __int8)*(_QWORD *)v18 ^ 6) & 6) == 0 && *(_DWORD *)(v18 + 8) > 3u )
      {
        v18 += 16;
        if ( i == v18 )
          goto LABEL_16;
      }
      v20 = (__int64 *)(*(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL);
      if ( v20 != (__int64 *)(*(_QWORD *)(a1 + 72) + 328LL) && v17 + *(_DWORD *)(v18 + 12) > v6 )
      {
        v28 = i;
        v31 = v6;
        v21 = sub_35E8960(a1, *v20);
        v6 = v31;
        i = v28;
        if ( v17 < v21 )
          return (unsigned int)dword_5040368;
        v22 = v17 + *(_DWORD *)(v18 + 12) - v31 - v21;
        if ( (int)v8 < v22 )
          v8 = v22;
      }
    }
LABEL_16:
    if ( !v7 )
      BUG();
    if ( (*(_BYTE *)v7 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
        v7 = *(_QWORD *)(v7 + 8);
    }
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v26 != v7 );
  return v8;
}
