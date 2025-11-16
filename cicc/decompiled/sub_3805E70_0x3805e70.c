// Function: sub_3805E70
// Address: 0x3805e70
//
unsigned __int64 __fastcall sub_3805E70(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v5; // eax
  char v6; // dl
  __int64 v7; // rdi
  int v8; // ecx
  unsigned int v9; // esi
  int *v10; // r14
  int v11; // r8d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  int v15; // ecx
  int v16; // eax
  unsigned int v17; // edi
  __int64 v18; // rdx
  int v19; // r9d
  __int64 v21; // rcx
  __int64 v22; // rcx
  int v23; // r9d
  __int64 v24; // rdx
  int v25; // edx
  int v26; // r10d

  v5 = sub_375D5B0(a1, a2, a3);
  v6 = *(_BYTE *)(a1 + 912) & 1;
  if ( v6 )
  {
    v7 = a1 + 920;
    v8 = 7;
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 928);
    v7 = *(_QWORD *)(a1 + 920);
    if ( !(_DWORD)v21 )
      goto LABEL_17;
    v8 = v21 - 1;
  }
  v9 = v8 & (37 * v5);
  v10 = (int *)(v7 + 8LL * v9);
  v11 = *v10;
  if ( v5 == *v10 )
    goto LABEL_4;
  v23 = 1;
  while ( v11 != -1 )
  {
    v9 = v8 & (v23 + v9);
    v10 = (int *)(v7 + 8LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
      goto LABEL_4;
    ++v23;
  }
  if ( v6 )
  {
    v22 = 64;
    goto LABEL_18;
  }
  v21 = *(unsigned int *)(a1 + 928);
LABEL_17:
  v22 = 8 * v21;
LABEL_18:
  v10 = (int *)(v7 + v22);
LABEL_4:
  v12 = 64;
  if ( !v6 )
    v12 = 8LL * *(unsigned int *)(a1 + 928);
  if ( v10 != (int *)(v7 + v12) )
  {
    sub_37593F0(a1, v10 + 1);
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v14 = a1 + 520;
      v15 = 7;
    }
    else
    {
      v13 = *(unsigned int *)(a1 + 528);
      v14 = *(_QWORD *)(a1 + 520);
      if ( !(_DWORD)v13 )
        goto LABEL_24;
      v15 = v13 - 1;
    }
    v16 = v10[1];
    v17 = v15 & (37 * v16);
    v18 = v14 + 24LL * v17;
    v19 = *(_DWORD *)v18;
    if ( v16 == *(_DWORD *)v18 )
      return *(_QWORD *)(v18 + 8);
    v25 = 1;
    while ( v19 != -1 )
    {
      v26 = v25 + 1;
      v17 = v15 & (v25 + v17);
      v18 = v14 + 24LL * v17;
      v19 = *(_DWORD *)v18;
      if ( v16 == *(_DWORD *)v18 )
        return *(_QWORD *)(v18 + 8);
      v25 = v26;
    }
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v24 = 192;
      goto LABEL_25;
    }
    v13 = *(unsigned int *)(a1 + 528);
LABEL_24:
    v24 = 24 * v13;
LABEL_25:
    v18 = v14 + v24;
    return *(_QWORD *)(v18 + 8);
  }
  return a2;
}
