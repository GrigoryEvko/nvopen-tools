// Function: sub_B93380
// Address: 0xb93380
//
unsigned __int64 __fastcall sub_B93380(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 *v3; // rdx
  unsigned int v4; // eax
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v14; // rax
  __int64 *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 *v19; // rdx
  unsigned int v20; // eax
  __int64 v21; // rcx
  int v22; // [rsp+Ch] [rbp-24h] BYREF
  __int64 v23; // [rsp+10h] [rbp-20h] BYREF
  __int64 v24[3]; // [rsp+18h] [rbp-18h] BYREF

  v1 = *(_QWORD *)(a1 + 56);
  v22 = 0;
  v23 = v1;
  if ( v1 && *(_BYTE *)v1 == 1 )
  {
    v2 = *(_QWORD *)(v1 + 136);
    v3 = *(__int64 **)(v2 + 24);
    v4 = *(_DWORD *)(v2 + 32);
    if ( v4 > 0x40 )
    {
      v5 = *v3;
    }
    else
    {
      v5 = 0;
      if ( v4 )
        v5 = (__int64)((_QWORD)v3 << (64 - (unsigned __int8)v4)) >> (64 - (unsigned __int8)v4);
    }
    v24[0] = v5;
    v22 = sub_AF6E60(&v22, v24);
    v6 = *(_QWORD *)(a1 + 64);
    v23 = v6;
    if ( !v6 )
      goto LABEL_7;
  }
  else
  {
    v22 = sub_AF7970(&v22, &v23);
    v6 = *(_QWORD *)(a1 + 64);
    v23 = v6;
    if ( !v6 )
      goto LABEL_7;
  }
  if ( *(_BYTE *)v6 == 1 )
  {
    v14 = *(_QWORD *)(v6 + 136);
    v15 = *(__int64 **)(v14 + 24);
    v16 = *(_DWORD *)(v14 + 32);
    if ( v16 > 0x40 )
    {
      v17 = *v15;
    }
    else
    {
      v17 = 0;
      if ( v16 )
        v17 = (__int64)((_QWORD)v15 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
    }
    v24[0] = v17;
    v22 = sub_AF6E60(&v22, v24);
    v7 = *(_QWORD *)(a1 + 72);
    v23 = v7;
    if ( !v7 )
      goto LABEL_21;
    goto LABEL_8;
  }
LABEL_7:
  v22 = sub_AF7970(&v22, &v23);
  v7 = *(_QWORD *)(a1 + 72);
  v23 = v7;
  if ( !v7 )
    goto LABEL_21;
LABEL_8:
  if ( *(_BYTE *)v7 == 1 )
  {
    v8 = *(_QWORD *)(v7 + 136);
    v9 = *(__int64 **)(v8 + 24);
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
    {
      v11 = *v9;
    }
    else
    {
      v11 = 0;
      if ( v10 )
        v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
    }
    v24[0] = v11;
    v22 = sub_AF6E60(&v22, v24);
    v12 = *(_QWORD *)(a1 + 80);
    v23 = v12;
    if ( !v12 )
      goto LABEL_13;
    goto LABEL_22;
  }
LABEL_21:
  v22 = sub_AF7970(&v22, &v23);
  v12 = *(_QWORD *)(a1 + 80);
  v23 = v12;
  if ( !v12 )
  {
LABEL_13:
    v22 = sub_AF7970(&v22, &v23);
    return sub_AF95C0(
             &v22,
             (__int64 *)a1,
             (__int64 *)(a1 + 8),
             (int *)(a1 + 16),
             (__int64 *)(a1 + 24),
             (__int64 *)(a1 + 48),
             (int *)(a1 + 44));
  }
LABEL_22:
  if ( *(_BYTE *)v12 != 1 )
    goto LABEL_13;
  v18 = *(_QWORD *)(v12 + 136);
  v19 = *(__int64 **)(v18 + 24);
  v20 = *(_DWORD *)(v18 + 32);
  if ( v20 > 0x40 )
  {
    v21 = *v19;
  }
  else
  {
    v21 = 0;
    if ( v20 )
      v21 = (__int64)((_QWORD)v19 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
  }
  v24[0] = v21;
  v22 = sub_AF6E60(&v22, v24);
  return sub_AF95C0(
           &v22,
           (__int64 *)a1,
           (__int64 *)(a1 + 8),
           (int *)(a1 + 16),
           (__int64 *)(a1 + 24),
           (__int64 *)(a1 + 48),
           (int *)(a1 + 44));
}
