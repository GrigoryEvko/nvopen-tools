// Function: sub_2650CE0
// Address: 0x2650ce0
//
bool __fastcall sub_2650CE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r15
  __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // rcx
  unsigned int v10; // r8d
  __int64 v11; // rdi
  __int64 *v12; // r10
  int v13; // r11d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r9
  unsigned int v17; // r9d
  __int64 *v18; // r13
  __int64 v19; // rcx
  int v20; // r15d
  __int64 *v21; // r10
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r14
  int v26; // eax
  int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rdx
  int v30; // edx
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD v34[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (__int64 *)(a2 + 32);
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 24);
  if ( !v8 )
  {
    v34[0] = 0;
    ++*(_QWORD *)v7;
LABEL_36:
    v8 *= 2;
    goto LABEL_37;
  }
  v9 = *(_QWORD *)(a2 + 32);
  v10 = v8 - 1;
  v11 = *(_QWORD *)(v7 + 8);
  v12 = 0;
  v13 = 1;
  v14 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v9 == *v15 )
  {
LABEL_3:
    v17 = *((_DWORD *)v15 + 2);
    v18 = (__int64 *)(a3 + 32);
    goto LABEL_4;
  }
  while ( v16 != -4096 )
  {
    if ( !v12 && v16 == -8192 )
      v12 = v15;
    v14 = v10 & (v13 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v9 == *v15 )
      goto LABEL_3;
    ++v13;
  }
  if ( !v12 )
    v12 = v15;
  v34[0] = v12;
  v26 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v27 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v8 )
    goto LABEL_36;
  if ( v8 - *(_DWORD *)(v7 + 20) - v27 <= v8 >> 3 )
  {
LABEL_37:
    sub_D1FCE0(v7, v8);
    sub_264A5C0(v7, v3, v34);
    v27 = *(_DWORD *)(v7 + 16) + 1;
  }
  *(_DWORD *)(v7 + 16) = v27;
  v28 = v34[0];
  if ( *(_QWORD *)v34[0] != -4096 )
    --*(_DWORD *)(v7 + 20);
  v29 = *(_QWORD *)(a2 + 32);
  *(_DWORD *)(v28 + 8) = 0;
  v18 = (__int64 *)(a3 + 32);
  *(_QWORD *)v28 = v29;
  v7 = *a1;
  v8 = *(_DWORD *)(*a1 + 24);
  if ( !v8 )
  {
    v34[0] = 0;
    ++*(_QWORD *)v7;
    goto LABEL_20;
  }
  v11 = *(_QWORD *)(v7 + 8);
  v10 = v8 - 1;
  v17 = 0;
LABEL_4:
  v19 = *(_QWORD *)(a3 + 32);
  v20 = 1;
  v21 = 0;
  v22 = v10 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v23 = (__int64 *)(v11 + 16LL * v22);
  v24 = *v23;
  if ( *v23 == v19 )
    return v17 < *((_DWORD *)v23 + 2);
  while ( v24 != -4096 )
  {
    if ( !v21 && v24 == -8192 )
      v21 = v23;
    v22 = v10 & (v20 + v22);
    v23 = (__int64 *)(v11 + 16LL * v22);
    v24 = *v23;
    if ( v19 == *v23 )
      return v17 < *((_DWORD *)v23 + 2);
    ++v20;
  }
  if ( !v21 )
    v21 = v23;
  v34[0] = v21;
  v31 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v30 = v31 + 1;
  if ( 4 * (v31 + 1) < 3 * v8 )
  {
    if ( v8 - *(_DWORD *)(v7 + 20) - v30 > v8 >> 3 )
      goto LABEL_32;
    goto LABEL_21;
  }
LABEL_20:
  v8 *= 2;
LABEL_21:
  sub_D1FCE0(v7, v8);
  sub_264A5C0(v7, v18, v34);
  v30 = *(_DWORD *)(v7 + 16) + 1;
LABEL_32:
  *(_DWORD *)(v7 + 16) = v30;
  v32 = v34[0];
  if ( *(_QWORD *)v34[0] != -4096 )
    --*(_DWORD *)(v7 + 20);
  v33 = *(_QWORD *)(a3 + 32);
  *(_DWORD *)(v32 + 8) = 0;
  *(_QWORD *)v32 = v33;
  return 0;
}
