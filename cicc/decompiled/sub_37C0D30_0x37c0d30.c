// Function: sub_37C0D30
// Address: 0x37c0d30
//
bool __fastcall sub_37C0D30(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // r12
  unsigned int v9; // edi
  __int64 v10; // rcx
  int v11; // r15d
  __int64 *v12; // r9
  unsigned int v13; // r8d
  __int64 *v14; // rax
  __int64 v15; // r11
  unsigned int v16; // r9d
  __int64 v17; // rax
  int v18; // r14d
  __int64 *v19; // r10
  unsigned int v20; // r8d
  __int64 *v21; // rdx
  __int64 v22; // r11
  int v24; // eax
  int v25; // ecx
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // [rsp+8h] [rbp-48h] BYREF
  __int64 v29; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *a1;
  v6 = *a2;
  v7 = *(_DWORD *)(*a1 + 688);
  v29 = v6;
  v8 = v5 + 664;
  if ( !v7 )
  {
    v30[0] = 0;
    ++*(_QWORD *)(v5 + 664);
LABEL_36:
    v7 *= 2;
    goto LABEL_37;
  }
  v9 = v7 - 1;
  v10 = *(_QWORD *)(v5 + 672);
  v11 = 1;
  v12 = 0;
  v13 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v6 == *v14 )
  {
LABEL_3:
    v16 = *((_DWORD *)v14 + 2);
    v17 = **(_QWORD **)(a3 + 80);
    v28 = v17;
    goto LABEL_4;
  }
  while ( v15 != -4096 )
  {
    if ( v15 == -8192 && !v12 )
      v12 = v14;
    v13 = v9 & (v11 + v13);
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v6 == *v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v30[0] = v12;
  v24 = *(_DWORD *)(v5 + 680);
  ++*(_QWORD *)(v5 + 664);
  v25 = v24 + 1;
  if ( 4 * (v24 + 1) >= 3 * v7 )
    goto LABEL_36;
  if ( v7 - *(_DWORD *)(v5 + 684) - v25 <= v7 >> 3 )
  {
LABEL_37:
    sub_2E515B0(v5 + 664, v7);
    sub_2E50510(v5 + 664, &v29, v30);
    v6 = v29;
    v12 = (__int64 *)v30[0];
    v25 = *(_DWORD *)(v5 + 680) + 1;
  }
  *(_DWORD *)(v5 + 680) = v25;
  if ( *v12 != -4096 )
    --*(_DWORD *)(v5 + 684);
  *v12 = v6;
  *((_DWORD *)v12 + 2) = 0;
  v5 = *a1;
  v7 = *(_DWORD *)(*a1 + 688);
  v10 = *(_QWORD *)(*a1 + 672);
  v8 = *a1 + 664;
  v17 = **(_QWORD **)(a3 + 80);
  v28 = v17;
  if ( !v7 )
  {
    v30[0] = 0;
    ++*(_QWORD *)(v5 + 664);
    goto LABEL_20;
  }
  v16 = 0;
  v9 = v7 - 1;
LABEL_4:
  v18 = 1;
  v19 = 0;
  v20 = v9 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v21 = (__int64 *)(v10 + 16LL * v20);
  v22 = *v21;
  if ( *v21 == v17 )
    return v16 < *((_DWORD *)v21 + 2);
  while ( v22 != -4096 )
  {
    if ( !v19 && v22 == -8192 )
      v19 = v21;
    v20 = v9 & (v18 + v20);
    v21 = (__int64 *)(v10 + 16LL * v20);
    v22 = *v21;
    if ( *v21 == v17 )
      return v16 < *((_DWORD *)v21 + 2);
    ++v18;
  }
  if ( !v19 )
    v19 = v21;
  v30[0] = v19;
  v27 = *(_DWORD *)(v5 + 680);
  ++*(_QWORD *)(v5 + 664);
  v26 = v27 + 1;
  if ( 4 * v26 < 3 * v7 )
  {
    if ( v7 - *(_DWORD *)(v5 + 684) - v26 > v7 >> 3 )
      goto LABEL_32;
    goto LABEL_21;
  }
LABEL_20:
  v7 *= 2;
LABEL_21:
  sub_2E515B0(v8, v7);
  sub_2E50510(v8, &v28, v30);
  v17 = v28;
  v19 = (__int64 *)v30[0];
  v26 = *(_DWORD *)(v5 + 680) + 1;
LABEL_32:
  *(_DWORD *)(v5 + 680) = v26;
  if ( *v19 != -4096 )
    --*(_DWORD *)(v5 + 684);
  *v19 = v17;
  *((_DWORD *)v19 + 2) = 0;
  return 0;
}
