// Function: sub_C1C670
// Address: 0xc1c670
//
__int64 __fastcall sub_C1C670(__int64 a1, __int64 a2, unsigned __int64 **a3)
{
  int v4; // r12d
  size_t v6; // r14
  const void *v7; // r15
  __int64 v8; // r8
  size_t v9; // rcx
  int v10; // r12d
  int v11; // r11d
  unsigned int v12; // r14d
  bool v13; // r9
  __int64 v14; // r10
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  int v18; // eax
  unsigned int v19; // r14d
  __int64 v20; // [rsp+0h] [rbp-100h]
  int v21; // [rsp+Ch] [rbp-F4h]
  __int64 v22; // [rsp+10h] [rbp-F0h]
  __int64 v23; // [rsp+10h] [rbp-F0h]
  size_t v24; // [rsp+18h] [rbp-E8h]
  __int64 v25; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE v26[208]; // [rsp+30h] [rbp-D0h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(const void **)a2;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v6;
  if ( *(_QWORD *)a2 )
  {
    v22 = *(_QWORD *)(a1 + 8);
    sub_C7D030(v26);
    sub_C7D280(v26, v7, v6);
    sub_C7D290(v26, &v25);
    LODWORD(v6) = v25;
    v7 = *(const void **)a2;
    v9 = *(_QWORD *)(a2 + 8);
    v8 = v22;
  }
  v10 = v4 - 1;
  v11 = 1;
  v12 = v10 & v6;
  v13 = v7 == 0;
  v14 = 0;
  while ( 1 )
  {
    v15 = (unsigned __int64 *)(v8 + 16LL * v12);
    v16 = v15[1];
    v17 = *v15;
    if ( v16 != v9 )
      break;
    if ( (const void *)v17 == v7 )
      goto LABEL_17;
    if ( !v17 || v13 )
      break;
    v20 = v8;
    v21 = v11;
    v23 = v14;
    v24 = v9;
    v18 = memcmp(v7, (const void *)v17, v9);
    v9 = v24;
    v14 = v23;
    v11 = v21;
    v8 = v20;
    v13 = 0;
    if ( !v18 )
    {
LABEL_17:
      *a3 = v15;
      return 1;
    }
LABEL_16:
    v19 = v11 + v12;
    ++v11;
    v12 = v10 & v19;
  }
  if ( v16 != -1 )
  {
    if ( !(v14 | v17) && v16 == -2 )
      v14 = v8 + 16LL * v12;
    goto LABEL_16;
  }
  if ( v17 )
    goto LABEL_16;
  if ( !v14 )
    v14 = v8 + 16LL * v12;
  *a3 = (unsigned __int64 *)v14;
  return 0;
}
