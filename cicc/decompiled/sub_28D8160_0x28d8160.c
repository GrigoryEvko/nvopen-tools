// Function: sub_28D8160
// Address: 0x28d8160
//
bool __fastcall sub_28D8160(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r12
  __int64 v7; // r8
  unsigned int v8; // ecx
  int v9; // r14d
  __int64 *v10; // r9
  __int64 v11; // rdx
  unsigned int v12; // edi
  __int64 *v13; // rax
  __int64 v14; // r10
  unsigned int v15; // r9d
  __int64 v16; // r8
  int v17; // r14d
  __int64 *v18; // r10
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // r11
  int v23; // eax
  int v24; // edx
  int v25; // edx
  int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-40h] BYREF
  __int64 v28; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v29[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v28 = a2;
  v27 = a3;
  v5 = *(_DWORD *)(v4 + 1384);
  v6 = v4 + 1360;
  if ( !v5 )
  {
    v29[0] = 0;
    ++*(_QWORD *)(v4 + 1360);
LABEL_36:
    v5 *= 2;
    goto LABEL_37;
  }
  v7 = v28;
  v8 = v5 - 1;
  v9 = 1;
  v10 = 0;
  v11 = *(_QWORD *)(v4 + 1368);
  v12 = (v5 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
  v13 = (__int64 *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( v28 == *v13 )
  {
LABEL_3:
    v15 = *((_DWORD *)v13 + 2);
    goto LABEL_4;
  }
  while ( v14 != -4096 )
  {
    if ( !v10 && v14 == -8192 )
      v10 = v13;
    v12 = v8 & (v9 + v12);
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( v28 == *v13 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v13;
  v29[0] = v10;
  v23 = *(_DWORD *)(v4 + 1376);
  ++*(_QWORD *)(v4 + 1360);
  v24 = v23 + 1;
  if ( 4 * (v23 + 1) >= 3 * v5 )
    goto LABEL_36;
  if ( v5 - *(_DWORD *)(v4 + 1380) - v24 <= v5 >> 3 )
  {
LABEL_37:
    sub_CE3370(v4 + 1360, v5);
    sub_28CD4F0(v4 + 1360, &v28, v29);
    v7 = v28;
    v10 = (__int64 *)v29[0];
    v24 = *(_DWORD *)(v4 + 1376) + 1;
  }
  *(_DWORD *)(v4 + 1376) = v24;
  if ( *v10 != -4096 )
    --*(_DWORD *)(v4 + 1380);
  *v10 = v7;
  *((_DWORD *)v10 + 2) = 0;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 1384);
  v6 = *a1 + 1360;
  if ( !v5 )
  {
    v29[0] = 0;
    ++*(_QWORD *)(v4 + 1360);
    goto LABEL_20;
  }
  v11 = *(_QWORD *)(v4 + 1368);
  v8 = v5 - 1;
  v15 = 0;
LABEL_4:
  v16 = v27;
  v17 = 1;
  v18 = 0;
  v19 = v8 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
  v20 = (__int64 *)(v11 + 16LL * v19);
  v21 = *v20;
  if ( *v20 == v27 )
    return v15 < *((_DWORD *)v20 + 2);
  while ( v21 != -4096 )
  {
    if ( !v18 && v21 == -8192 )
      v18 = v20;
    v19 = v8 & (v17 + v19);
    v20 = (__int64 *)(v11 + 16LL * v19);
    v21 = *v20;
    if ( v27 == *v20 )
      return v15 < *((_DWORD *)v20 + 2);
    ++v17;
  }
  if ( !v18 )
    v18 = v20;
  v29[0] = v18;
  v26 = *(_DWORD *)(v4 + 1376);
  ++*(_QWORD *)(v4 + 1360);
  v25 = v26 + 1;
  if ( 4 * (v26 + 1) < 3 * v5 )
  {
    if ( v5 - *(_DWORD *)(v4 + 1380) - v25 > v5 >> 3 )
      goto LABEL_32;
    goto LABEL_21;
  }
LABEL_20:
  v5 *= 2;
LABEL_21:
  sub_CE3370(v6, v5);
  sub_28CD4F0(v6, &v27, v29);
  v16 = v27;
  v18 = (__int64 *)v29[0];
  v25 = *(_DWORD *)(v4 + 1376) + 1;
LABEL_32:
  *(_DWORD *)(v4 + 1376) = v25;
  if ( *v18 != -4096 )
    --*(_DWORD *)(v4 + 1380);
  *v18 = v16;
  *((_DWORD *)v18 + 2) = 0;
  return 0;
}
