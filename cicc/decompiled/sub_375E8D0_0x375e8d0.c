// Function: sub_375E8D0
// Address: 0x375e8d0
//
__int64 __fastcall sub_375E8D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int *v8; // r12
  char v9; // si
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // r9d
  __int64 v13; // rax
  int v14; // r8d
  __int64 v15; // r8
  int v16; // ecx
  int v17; // edx
  unsigned int v18; // edi
  __int64 v19; // rax
  int v20; // r9d
  __int64 result; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  int v27; // eax
  int v28; // r10d
  int v29; // r10d
  int v30[9]; // [rsp+Ch] [rbp-24h] BYREF

  v30[0] = sub_375D5B0(a1, a2, a3);
  v8 = sub_375C600(a1 + 1336, v30);
  sub_37593F0(a1, v8);
  v9 = *(_BYTE *)(a1 + 512) & 1;
  if ( v9 )
  {
    v10 = a1 + 520;
    v11 = 7;
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 528);
    v10 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v22 )
      goto LABEL_13;
    v11 = v22 - 1;
  }
  v12 = v11 & (37 * *v8);
  v13 = v10 + 24LL * v12;
  v14 = *(_DWORD *)v13;
  if ( *v8 == *(_DWORD *)v13 )
    goto LABEL_4;
  v26 = 1;
  while ( v14 != -1 )
  {
    v28 = v26 + 1;
    v12 = v11 & (v26 + v12);
    v13 = v10 + 24LL * v12;
    v14 = *(_DWORD *)v13;
    if ( *v8 == *(_DWORD *)v13 )
      goto LABEL_4;
    v26 = v28;
  }
  if ( v9 )
  {
    v24 = 192;
    goto LABEL_14;
  }
  v22 = *(unsigned int *)(a1 + 528);
LABEL_13:
  v24 = 24 * v22;
LABEL_14:
  v13 = v10 + v24;
LABEL_4:
  *(_QWORD *)a4 = *(_QWORD *)(v13 + 8);
  *(_DWORD *)(a4 + 8) = *(_DWORD *)(v13 + 16);
  sub_37593F0(a1, v8 + 1);
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v15 = a1 + 520;
    v16 = 7;
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 528);
    v15 = *(_QWORD *)(a1 + 520);
    if ( !(_DWORD)v23 )
      goto LABEL_16;
    v16 = v23 - 1;
  }
  v17 = v8[1];
  v18 = v16 & (37 * v17);
  v19 = v15 + 24LL * v18;
  v20 = *(_DWORD *)v19;
  if ( v17 != *(_DWORD *)v19 )
  {
    v27 = 1;
    while ( v20 != -1 )
    {
      v29 = v27 + 1;
      v18 = v16 & (v27 + v18);
      v19 = v15 + 24LL * v18;
      v20 = *(_DWORD *)v19;
      if ( v17 == *(_DWORD *)v19 )
        goto LABEL_7;
      v27 = v29;
    }
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v25 = 192;
      goto LABEL_17;
    }
    v23 = *(unsigned int *)(a1 + 528);
LABEL_16:
    v25 = 24 * v23;
LABEL_17:
    v19 = v15 + v25;
  }
LABEL_7:
  *(_QWORD *)a5 = *(_QWORD *)(v19 + 8);
  result = *(unsigned int *)(v19 + 16);
  *(_DWORD *)(a5 + 8) = result;
  return result;
}
