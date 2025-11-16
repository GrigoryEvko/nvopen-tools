// Function: sub_1E13870
// Address: 0x1e13870
//
unsigned __int64 __fastcall sub_1E13870(__int64 *a1, int a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  char v7; // r10
  int v8; // r8d
  bool v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r13
  _QWORD *v14; // rdx
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int8 v21; // [rsp+Eh] [rbp-52h]
  char v22; // [rsp+Fh] [rbp-51h]
  __int64 v23; // [rsp+10h] [rbp-50h]
  unsigned __int16 v24; // [rsp+2Dh] [rbp-33h]

  v4 = a1[2];
  if ( a1[3] != v4 )
  {
    v5 = *a1;
    v7 = 0;
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
LABEL_3:
      if ( *(_BYTE *)v4 || a2 != *(_DWORD *)(v4 + 8) )
      {
        v16 = a1[2];
        goto LABEL_12;
      }
      if ( a3 )
      {
        v10 = *(unsigned int *)(a3 + 8);
        v11 = (v4 - *(_QWORD *)(v5 + 32)) >> 3;
        v12 = *(_QWORD *)(v4 + 16);
        v13 = (unsigned int)(-858993459 * v11);
        if ( (unsigned int)v10 >= *(_DWORD *)(a3 + 12) )
        {
          v21 = v8;
          v22 = v7;
          v23 = *(_QWORD *)(v4 + 16);
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v8, -858993459);
          v8 = v21;
          v7 = v22;
          v12 = v23;
          v10 = *(unsigned int *)(a3 + 8);
        }
        v14 = (_QWORD *)(*(_QWORD *)a3 + 16 * v10);
        *v14 = v12;
        v14[1] = v13;
        ++*(_DWORD *)(a3 + 8);
        v5 = *a1;
      }
      v15 = *(_BYTE *)(v4 + 3) & 0x10;
      if ( (*(_BYTE *)(v4 + 4) & 1) != 0 || (*(_BYTE *)(v4 + 4) & 2) != 0 )
        break;
      if ( !v15 )
      {
        v16 = a1[2];
        v8 = 1;
        if ( v9 )
          goto LABEL_12;
LABEL_23:
        v20 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 32LL)
            + 0xFFFFFFF800000008LL * (unsigned int)((v16 - *(_QWORD *)(v5 + 32)) >> 3);
        if ( !*(_BYTE *)v20 && (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
          v9 = (*(_WORD *)(v20 + 2) & 0xFF0) != 0;
        goto LABEL_12;
      }
      v7 = 1;
      v16 = a1[2];
      if ( ((*(_DWORD *)v4 >> 8) & 0xFFF) != 0 )
      {
        v9 = 1;
        v8 = 1;
      }
LABEL_12:
      v4 = v16 + 40;
      a1[2] = v16 + 40;
      if ( v16 + 40 == a1[3] )
      {
        v17 = a1[1];
        while ( 1 )
        {
          v5 = *(_QWORD *)(v5 + 8);
          *a1 = v5;
          if ( v5 == v17 || (*(_BYTE *)(v5 + 46) & 4) == 0 )
            break;
          v4 = *(_QWORD *)(v5 + 32);
          a1[2] = v4;
          v18 = v4 + 40LL * *(unsigned int *)(v5 + 40);
          a1[3] = v18;
          if ( v4 != v18 )
            goto LABEL_3;
        }
        v4 = a1[2];
        if ( a1[3] == v4 )
          goto LABEL_18;
      }
    }
    if ( v15 )
    {
      v16 = a1[2];
      v7 = 1;
      goto LABEL_12;
    }
    v16 = a1[2];
    if ( v9 )
      goto LABEL_12;
    goto LABEL_23;
  }
  v7 = 0;
  LOBYTE(v8) = 0;
  v9 = 0;
LABEL_18:
  LOBYTE(v24) = v8;
  HIBYTE(v24) = v7;
  return v24 | ((unsigned __int64)v9 << 16);
}
