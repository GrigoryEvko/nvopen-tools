// Function: sub_19F52B0
// Address: 0x19f52b0
//
bool __fastcall sub_19F52B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // r8
  unsigned int v8; // ecx
  __int64 v9; // rdx
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // r10
  unsigned int v13; // r10d
  __int64 v14; // r8
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // r11
  int v19; // r14d
  __int64 *v20; // r9
  int v21; // eax
  int v22; // edx
  __int64 *v23; // r9
  int v24; // edx
  int v25; // r12d
  int v26; // eax
  __int64 *v27; // r14
  __int64 v28; // [rsp+0h] [rbp-40h] BYREF
  __int64 v29; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v30[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v29 = a2;
  v28 = a3;
  v5 = *(_DWORD *)(v4 + 1424);
  v6 = v4 + 1400;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 1400);
LABEL_19:
    v5 *= 2;
    goto LABEL_20;
  }
  v7 = v29;
  v8 = v5 - 1;
  v9 = *(_QWORD *)(v4 + 1408);
  v10 = (v5 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( v29 == *v11 )
  {
LABEL_3:
    v13 = *((_DWORD *)v11 + 2);
    goto LABEL_4;
  }
  v19 = 1;
  v20 = 0;
  while ( v12 != -8 )
  {
    if ( !v20 && v12 == -16 )
      v20 = v11;
    v10 = v8 & (v19 + v10);
    v11 = (__int64 *)(v9 + 16LL * v10);
    v12 = *v11;
    if ( v29 == *v11 )
      goto LABEL_3;
    ++v19;
  }
  if ( !v20 )
    v20 = v11;
  v21 = *(_DWORD *)(v4 + 1416);
  ++*(_QWORD *)(v4 + 1400);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v5 )
    goto LABEL_19;
  if ( v5 - *(_DWORD *)(v4 + 1420) - v22 <= v5 >> 3 )
  {
LABEL_20:
    sub_19F5120(v4 + 1400, v5);
    sub_19E6B80(v4 + 1400, &v29, v30);
    v20 = (__int64 *)v30[0];
    v7 = v29;
    v22 = *(_DWORD *)(v4 + 1416) + 1;
  }
  *(_DWORD *)(v4 + 1416) = v22;
  if ( *v20 != -8 )
    --*(_DWORD *)(v4 + 1420);
  *v20 = v7;
  *((_DWORD *)v20 + 2) = 0;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 1424);
  v6 = *a1 + 1400;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 1400);
LABEL_16:
    v5 *= 2;
LABEL_17:
    sub_19F5120(v6, v5);
    sub_19E6B80(v6, &v28, v30);
    v23 = (__int64 *)v30[0];
    v14 = v28;
    v24 = *(_DWORD *)(v4 + 1416) + 1;
    goto LABEL_28;
  }
  v9 = *(_QWORD *)(v4 + 1408);
  v8 = v5 - 1;
  v13 = 0;
LABEL_4:
  v14 = v28;
  v15 = v8 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v16 = (__int64 *)(v9 + 16LL * v15);
  v17 = *v16;
  if ( *v16 == v28 )
    return v13 < *((_DWORD *)v16 + 2);
  v25 = 1;
  v23 = 0;
  while ( v17 != -8 )
  {
    if ( v23 || v17 != -16 )
      v16 = v23;
    v15 = v8 & (v25 + v15);
    v27 = (__int64 *)(v9 + 16LL * v15);
    v17 = *v27;
    if ( v28 == *v27 )
      return v13 < *((_DWORD *)v27 + 2);
    ++v25;
    v23 = v16;
    v16 = (__int64 *)(v9 + 16LL * v15);
  }
  if ( !v23 )
    v23 = v16;
  v26 = *(_DWORD *)(v4 + 1416);
  ++*(_QWORD *)(v4 + 1400);
  v24 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v5 )
    goto LABEL_16;
  if ( v5 - *(_DWORD *)(v4 + 1420) - v24 <= v5 >> 3 )
    goto LABEL_17;
LABEL_28:
  *(_DWORD *)(v4 + 1416) = v24;
  if ( *v23 != -8 )
    --*(_DWORD *)(v4 + 1420);
  *v23 = v14;
  *((_DWORD *)v23 + 2) = 0;
  return 0;
}
