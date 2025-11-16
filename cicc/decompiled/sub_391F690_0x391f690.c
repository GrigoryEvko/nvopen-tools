// Function: sub_391F690
// Address: 0x391f690
//
__int64 __fastcall sub_391F690(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v6; // esi
  unsigned int v7; // r9d
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // rdi
  __int64 *v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r10
  __int64 v16; // r13
  int i; // r11d
  int v18; // r11d
  __int64 *v19; // r10
  int v20; // eax
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rdi
  int v26; // ecx
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // r11d
  __int64 *v31; // r9
  int v32; // eax
  __int64 v33; // rdi
  int v34; // ecx
  __int64 v35; // r8
  int v36; // r11d
  unsigned int v37; // eax
  __int64 v38; // rsi
  _QWORD v39[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v40[2]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v41; // [rsp-38h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( *(_DWORD *)(a2 + 24) != 6 )
    return *(unsigned int *)(v2 + 16);
  v6 = *(_DWORD *)(a1 + 120);
  if ( !v6 )
  {
LABEL_7:
    if ( (*(_BYTE *)v2 & 4) != 0 )
    {
      v12 = *(__int64 **)(v2 - 8);
      v13 = *v12;
      v14 = v12 + 2;
    }
    else
    {
      v13 = 0;
      v14 = 0;
    }
    v39[0] = v14;
    v41 = 1283;
    v40[0] = "symbol not found in type index space: ";
    v39[1] = v13;
    v40[1] = v39;
    sub_16BCFB0((__int64)v40, 1u);
  }
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a1 + 104);
  v9 = (v6 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == v2 )
    return *((unsigned int *)v10 + 2);
  v15 = *v10;
  LODWORD(v16) = (v6 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  for ( i = 1; ; ++i )
  {
    if ( v15 == -8 )
      goto LABEL_7;
    v16 = v7 & ((_DWORD)v16 + i);
    v15 = *(_QWORD *)(v8 + 16 * v16);
    if ( v15 == v2 )
      break;
  }
  v18 = 1;
  v19 = 0;
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v19 )
      v19 = v10;
    v9 = v7 & (v18 + v9);
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == v2 )
      return *((unsigned int *)v10 + 2);
    ++v18;
  }
  v20 = *(_DWORD *)(a1 + 112);
  v21 = a1 + 96;
  if ( !v19 )
    v19 = v10;
  ++*(_QWORD *)(a1 + 96);
  v22 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v6 )
  {
    sub_391E830(v21, 2 * v6);
    v24 = *(_DWORD *)(a1 + 120);
    if ( v24 )
    {
      v25 = *(_QWORD *)(a2 + 8);
      v26 = v24 - 1;
      v27 = *(_QWORD *)(a1 + 104);
      v28 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v22 = *(_DWORD *)(a1 + 112) + 1;
      v19 = (__int64 *)(v27 + 16LL * v28);
      v29 = *v19;
      if ( *v19 == v25 )
        goto LABEL_20;
      v30 = 1;
      v31 = 0;
      while ( v29 != -8 )
      {
        if ( v29 == -16 && !v31 )
          v31 = v19;
        v28 = v26 & (v30 + v28);
        v19 = (__int64 *)(v27 + 16LL * v28);
        v29 = *v19;
        if ( v25 == *v19 )
          goto LABEL_20;
        ++v30;
      }
LABEL_27:
      if ( v31 )
        v19 = v31;
      goto LABEL_20;
    }
LABEL_49:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 116) - v22 <= v6 >> 3 )
  {
    sub_391E830(v21, v6);
    v32 = *(_DWORD *)(a1 + 120);
    if ( v32 )
    {
      v33 = *(_QWORD *)(a2 + 8);
      v34 = v32 - 1;
      v35 = *(_QWORD *)(a1 + 104);
      v31 = 0;
      v36 = 1;
      v37 = (v32 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v22 = *(_DWORD *)(a1 + 112) + 1;
      v19 = (__int64 *)(v35 + 16LL * v37);
      v38 = *v19;
      if ( v33 == *v19 )
        goto LABEL_20;
      while ( v38 != -8 )
      {
        if ( v38 == -16 && !v31 )
          v31 = v19;
        v37 = v34 & (v36 + v37);
        v19 = (__int64 *)(v35 + 16LL * v37);
        v38 = *v19;
        if ( v33 == *v19 )
          goto LABEL_20;
        ++v36;
      }
      goto LABEL_27;
    }
    goto LABEL_49;
  }
LABEL_20:
  *(_DWORD *)(a1 + 112) = v22;
  if ( *v19 != -8 )
    --*(_DWORD *)(a1 + 116);
  v23 = *(_QWORD *)(a2 + 8);
  *((_DWORD *)v19 + 2) = 0;
  *v19 = v23;
  return 0;
}
