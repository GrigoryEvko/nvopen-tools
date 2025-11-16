// Function: sub_10700F0
// Address: 0x10700f0
//
__int64 __fastcall sub_10700F0(__int64 a1, __int64 a2)
{
  int v3; // r13d
  int v4; // edx
  int v5; // eax
  unsigned int v6; // r13d
  int v7; // eax
  int v8; // r12d
  int v9; // ecx
  unsigned int v10; // r12d
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rdi
  int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdi
  unsigned __int8 v28[52]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = *(_DWORD *)(a2 + 16);
  v4 = v3 & 0x7FFFFFFF;
  if ( (((v3 | 0x80000000) >> 24) & 0x80u) == 0 )
    v4 = 0;
  v5 = *(_DWORD *)(a2 + 12) << 8;
  if ( (((*(_DWORD *)(a2 + 12) | 0x80000000) >> 24) & 0x80u) == 0 )
    v5 = 0;
  v6 = v5 | v4 | (*(_DWORD *)(a2 + 8) << 16);
  v7 = *(_DWORD *)(a2 + 20);
  if ( v7
    || (*(_DWORD *)(a2 + 24) & 0x7FFFFFFF) != 0
    || (*(_DWORD *)(a2 + 28) & 0x7FFFFFFF) != 0
    || (v10 = *(_DWORD *)(a2 + 32) & 0x7FFFFFFF) != 0 )
  {
    v8 = 0;
    v9 = *(_DWORD *)(a2 + 28) & 0x7FFFFFFF;
    if ( *(char *)(a2 + 31) >= 0 )
      v9 = 0;
    if ( *(char *)(a2 + 27) < 0 )
      v8 = *(_DWORD *)(a2 + 24) << 8;
    v10 = v9 | (v7 << 16) | v8;
  }
  if ( *(_BYTE *)a2 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    v20 = 50;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v20 = 838860800;
    *(_DWORD *)v28 = v20;
    sub_CB6200(v19, v28, 4u);
    v21 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    v22 = 24;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v22 = 402653184;
    *(_DWORD *)v28 = v22;
    sub_CB6200(v21, v28, 4u);
    v23 = *(_DWORD *)(a2 + 4);
    v24 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v23 = _byteswap_ulong(v23);
    *(_DWORD *)v28 = v23;
    sub_CB6200(v24, v28, 4u);
    v25 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v6 = _byteswap_ulong(v6);
    *(_DWORD *)v28 = v6;
    sub_CB6200(v25, v28, 4u);
    v26 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v10 = _byteswap_ulong(v10);
    *(_DWORD *)v28 = v10;
    sub_CB6200(v26, v28, 4u);
    v27 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    *(_DWORD *)v28 = 0;
    return sub_CB6200(v27, v28, 4u);
  }
  else
  {
    v11 = *(unsigned int *)(a2 + 4);
    if ( (unsigned int)v11 > 3 )
      BUG();
    v12 = *(_DWORD *)&asc_3F8F230[4 * v11];
    v13 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v12 = _byteswap_ulong(v12);
    *(_DWORD *)v28 = v12;
    sub_CB6200(v13, v28, 4u);
    v14 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    v15 = 16;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v15 = 0x10000000;
    *(_DWORD *)v28 = v15;
    sub_CB6200(v14, v28, 4u);
    v16 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v6 = _byteswap_ulong(v6);
    *(_DWORD *)v28 = v6;
    sub_CB6200(v16, v28, 4u);
    v17 = *(_QWORD *)(*(_QWORD *)a1 + 2048LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 2056LL) != 1 )
      v10 = _byteswap_ulong(v10);
    *(_DWORD *)v28 = v10;
    return sub_CB6200(v17, v28, 4u);
  }
}
