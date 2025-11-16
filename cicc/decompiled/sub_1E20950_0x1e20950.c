// Function: sub_1E20950
// Address: 0x1e20950
//
bool __fastcall sub_1E20950(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  unsigned int v6; // edi
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r10
  unsigned int v11; // r11d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  int v16; // r11d
  __int64 *v17; // r9
  int v18; // eax
  int v19; // edx
  __int64 v20; // rax
  __int64 *v21; // r9
  int v22; // edx
  int v23; // r12d
  int v24; // eax
  __int64 v25; // rax
  __int64 *v26; // r13
  __int64 v27; // [rsp+0h] [rbp-40h] BYREF
  __int64 v28; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v29[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v28 = a2;
  v27 = a3;
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
LABEL_19:
    v5 *= 2;
    goto LABEL_20;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(v4 + 8);
  v8 = (v5 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( v28 == *v9 )
  {
LABEL_3:
    v11 = *((_DWORD *)v9 + 2);
    goto LABEL_4;
  }
  v16 = 1;
  v17 = 0;
  while ( v10 != -8 )
  {
    if ( !v17 && v10 == -16 )
      v17 = v9;
    v8 = v6 & (v16 + v8);
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v28 == *v9 )
      goto LABEL_3;
    ++v16;
  }
  if ( !v17 )
    v17 = v9;
  v18 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v5 )
    goto LABEL_19;
  if ( v5 - *(_DWORD *)(v4 + 20) - v19 <= v5 >> 3 )
  {
LABEL_20:
    sub_1E20790(v4, v5);
    sub_1E1F1A0(v4, &v28, v29);
    v17 = (__int64 *)v29[0];
    v19 = *(_DWORD *)(v4 + 16) + 1;
  }
  *(_DWORD *)(v4 + 16) = v19;
  if ( *v17 != -8 )
    --*(_DWORD *)(v4 + 20);
  v20 = v28;
  *((_DWORD *)v17 + 2) = 0;
  *v17 = v20;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
LABEL_16:
    v5 *= 2;
LABEL_17:
    sub_1E20790(v4, v5);
    sub_1E1F1A0(v4, &v27, v29);
    v21 = (__int64 *)v29[0];
    v22 = *(_DWORD *)(v4 + 16) + 1;
    goto LABEL_28;
  }
  v7 = *(_QWORD *)(v4 + 8);
  v6 = v5 - 1;
  v11 = 0;
LABEL_4:
  v12 = v6 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v13 = (__int64 *)(v7 + 16LL * v12);
  v14 = *v13;
  if ( *v13 == v27 )
    return v11 < *((_DWORD *)v13 + 2);
  v23 = 1;
  v21 = 0;
  while ( v14 != -8 )
  {
    if ( v21 || v14 != -16 )
      v13 = v21;
    v12 = v6 & (v23 + v12);
    v26 = (__int64 *)(v7 + 16LL * v12);
    v14 = *v26;
    if ( v27 == *v26 )
      return v11 < *((_DWORD *)v26 + 2);
    ++v23;
    v21 = v13;
    v13 = (__int64 *)(v7 + 16LL * v12);
  }
  if ( !v21 )
    v21 = v13;
  v24 = *(_DWORD *)(v4 + 16);
  ++*(_QWORD *)v4;
  v22 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v5 )
    goto LABEL_16;
  if ( v5 - *(_DWORD *)(v4 + 20) - v22 <= v5 >> 3 )
    goto LABEL_17;
LABEL_28:
  *(_DWORD *)(v4 + 16) = v22;
  if ( *v21 != -8 )
    --*(_DWORD *)(v4 + 20);
  v25 = v27;
  *((_DWORD *)v21 + 2) = 0;
  *v21 = v25;
  return 0;
}
