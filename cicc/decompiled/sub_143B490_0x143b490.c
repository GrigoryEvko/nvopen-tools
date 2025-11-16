// Function: sub_143B490
// Address: 0x143b490
//
char __fastcall sub_143B490(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // si
  __int64 v6; // rax
  int v7; // r10d
  unsigned int v8; // ecx
  __int64 v9; // rdx
  __int64 v10; // r11
  int v11; // r10d
  int v12; // ebx
  unsigned int v13; // r12d
  __int64 v14; // rcx
  __int64 v15; // r11
  __int64 v16; // rax
  char v17; // r10
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // rdx
  int v23; // edx
  int v24; // ebx

  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( v5 )
  {
    v6 = a1 + 16;
    v7 = 31;
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 24);
    v6 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v19 )
      goto LABEL_21;
    v7 = v19 - 1;
  }
  v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = v6 + 16LL * v8;
  v10 = *(_QWORD *)v9;
  if ( a2 == *(_QWORD *)v9 )
    goto LABEL_4;
  v23 = 1;
  while ( v10 != -8 )
  {
    v24 = v23 + 1;
    v8 = v7 & (v23 + v8);
    v9 = v6 + 16LL * v8;
    v10 = *(_QWORD *)v9;
    if ( a2 == *(_QWORD *)v9 )
      goto LABEL_4;
    v23 = v24;
  }
  if ( v5 )
  {
    v22 = 512;
    goto LABEL_22;
  }
  v19 = *(unsigned int *)(a1 + 24);
LABEL_21:
  v22 = 16 * v19;
LABEL_22:
  v9 = v6 + v22;
LABEL_4:
  if ( v5 )
  {
    v11 = 31;
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    v11 = v20 - 1;
    if ( !(_DWORD)v20 )
      goto LABEL_18;
  }
  v12 = 1;
  v13 = v11 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v14 = v6 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( a3 == *(_QWORD *)v14 )
    goto LABEL_7;
  while ( v15 != -8 )
  {
    v13 = v11 & (v12 + v13);
    v14 = v6 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( a3 == *(_QWORD *)v14 )
      goto LABEL_7;
    ++v12;
  }
  if ( v5 )
  {
    v21 = 512;
    goto LABEL_19;
  }
  v20 = *(unsigned int *)(a1 + 24);
LABEL_18:
  v21 = 16 * v20;
LABEL_19:
  v14 = v6 + v21;
LABEL_7:
  if ( v5 )
  {
    v16 = v6 + 512;
    if ( v9 != v16 )
    {
      if ( v14 == v16 )
        return 1;
      return *(_DWORD *)(v9 + 8) < *(_DWORD *)(v14 + 8);
    }
  }
  else
  {
    v16 = 16LL * *(unsigned int *)(a1 + 24) + v6;
    if ( v9 != v16 )
    {
      if ( v14 == v16 )
        return 1;
      return *(_DWORD *)(v9 + 8) < *(_DWORD *)(v14 + 8);
    }
  }
  v17 = 0;
  if ( v14 != v16 )
    return v17;
  return sub_143B0F0(a1, a2, a3);
}
