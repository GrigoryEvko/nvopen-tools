// Function: sub_261D3E0
// Address: 0x261d3e0
//
bool __fastcall sub_261D3E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  unsigned int v6; // edi
  __int64 v7; // rcx
  __int64 *v8; // r10
  int v9; // r11d
  unsigned int v10; // eax
  __int64 *v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r10d
  int v14; // r12d
  __int64 *v15; // r11
  unsigned int v16; // eax
  __int64 *v17; // r8
  __int64 v18; // r9
  int v20; // eax
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // edx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-40h] BYREF
  __int64 v29; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v30[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v29 = a2;
  v28 = a3;
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
  {
    v30[0] = 0;
    ++*(_QWORD *)v4;
LABEL_36:
    v5 *= 2;
    goto LABEL_37;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
  v11 = (__int64 *)(v7 + 40LL * v10);
  v12 = *v11;
  if ( v29 == *v11 )
  {
LABEL_3:
    v13 = *((_DWORD *)v11 + 2);
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = v6 & (v9 + v10);
    v11 = (__int64 *)(v7 + 40LL * v10);
    v12 = *v11;
    if ( v29 == *v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v30[0] = v8;
  v20 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v5 )
    goto LABEL_36;
  if ( v5 - *(_DWORD *)(v4 + 20) - v21 <= v5 >> 3 )
  {
LABEL_37:
    sub_261D190(v4, v5);
    sub_2618CC0(v4, &v29, v30);
    v21 = *(_DWORD *)(v4 + 16) + 1;
  }
  *(_DWORD *)(v4 + 16) = v21;
  v22 = v30[0];
  if ( *(_QWORD *)v30[0] != -4096 )
    --*(_DWORD *)(v4 + 20);
  v23 = v29;
  *(_OWORD *)(v22 + 8) = 0;
  *(_QWORD *)v22 = v23;
  *(_OWORD *)(v22 + 24) = 0;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
  {
    v30[0] = 0;
    ++*(_QWORD *)v4;
    goto LABEL_20;
  }
  v7 = *(_QWORD *)(v4 + 8);
  v6 = v5 - 1;
  v13 = 0;
LABEL_4:
  v14 = 1;
  v15 = 0;
  v16 = v6 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v17 = (__int64 *)(v7 + 40LL * v16);
  v18 = *v17;
  if ( *v17 == v28 )
    return v13 < *((_DWORD *)v17 + 2);
  while ( v18 != -4096 )
  {
    if ( !v15 && v18 == -8192 )
      v15 = v17;
    v16 = v6 & (v14 + v16);
    v17 = (__int64 *)(v7 + 40LL * v16);
    v18 = *v17;
    if ( v28 == *v17 )
      return v13 < *((_DWORD *)v17 + 2);
    ++v14;
  }
  if ( !v15 )
    v15 = v17;
  v30[0] = v15;
  v25 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v24 = v25 + 1;
  if ( 4 * (v25 + 1) < 3 * v5 )
  {
    if ( v5 - *(_DWORD *)(v4 + 20) - v24 > v5 >> 3 )
      goto LABEL_32;
    goto LABEL_21;
  }
LABEL_20:
  v5 *= 2;
LABEL_21:
  sub_261D190(v4, v5);
  sub_2618CC0(v4, &v28, v30);
  v24 = *(_DWORD *)(v4 + 16) + 1;
LABEL_32:
  *(_DWORD *)(v4 + 16) = v24;
  v26 = v30[0];
  if ( *(_QWORD *)v30[0] != -4096 )
    --*(_DWORD *)(v4 + 20);
  v27 = v28;
  *(_OWORD *)(v26 + 8) = 0;
  *(_QWORD *)v26 = v27;
  *(_OWORD *)(v26 + 24) = 0;
  return 0;
}
