// Function: sub_C77470
// Address: 0xc77470
//
__int16 __fastcall sub_C77470(__int64 a1, __int64 a2)
{
  unsigned int v4; // r13d
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v7; // edi
  __int64 v8; // rdx
  unsigned int v9; // edx
  unsigned int v10; // edi
  unsigned __int64 v11; // rax
  int v12; // eax
  __int16 result; // ax
  unsigned int v14; // esi
  unsigned __int64 v15; // rax
  unsigned int v16; // ebx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned int v19; // edi
  __int64 v20; // rdx
  int v21; // r12d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // [rsp+Ch] [rbp-64h]
  int v27; // [rsp+Ch] [rbp-64h]
  int v28; // [rsp+Ch] [rbp-64h]
  __int64 v29; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-58h]
  unsigned __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-48h]
  unsigned __int64 v33; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 8);
  v34 = v4;
  if ( v4 > 0x40 )
  {
    sub_C43780((__int64)&v33, (const void **)a1);
    v4 = v34;
    if ( v34 > 0x40 )
    {
      sub_C43D10((__int64)&v33);
      v4 = v34;
      v6 = v33;
      goto LABEL_5;
    }
    v5 = v33;
  }
  else
  {
    v5 = *(_QWORD *)a1;
  }
  v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v5;
  if ( !v4 )
    v6 = 0;
LABEL_5:
  v7 = *(_DWORD *)(a1 + 24);
  v32 = v4;
  v31 = v6;
  v8 = *(_QWORD *)(a1 + 16);
  if ( v7 > 0x40 )
    v8 = *(_QWORD *)(v8 + 8LL * ((v7 - 1) >> 6));
  if ( (v8 & (1LL << ((unsigned __int8)v7 - 1))) == 0 )
  {
    v23 = ~(1LL << ((unsigned __int8)v4 - 1));
    if ( v4 > 0x40 )
      *(_QWORD *)(v6 + 8LL * ((v4 - 1) >> 6)) &= v23;
    else
      v31 = v6 & v23;
  }
  v9 = *(_DWORD *)(a2 + 24);
  v34 = v9;
  if ( v9 > 0x40 )
  {
    sub_C43780((__int64)&v33, (const void **)(a2 + 16));
    v9 = v34;
  }
  else
  {
    v33 = *(_QWORD *)(a2 + 16);
  }
  v10 = *(_DWORD *)(a2 + 8);
  v11 = *(_QWORD *)a2;
  if ( v10 > 0x40 )
    v11 = *(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6));
  if ( (v11 & (1LL << ((unsigned __int8)v10 - 1))) == 0 )
  {
    v22 = 1LL << ((unsigned __int8)v9 - 1);
    if ( v9 <= 0x40 )
    {
      v33 |= v22;
      v12 = sub_C4C880((__int64)&v31, (__int64)&v33);
      goto LABEL_16;
    }
    *(_QWORD *)(v33 + 8LL * ((v9 - 1) >> 6)) |= v22;
    v9 = v34;
  }
  v26 = v9;
  v12 = sub_C4C880((__int64)&v31, (__int64)&v33);
  if ( v26 > 0x40 && v33 )
  {
    v27 = v12;
    j_j___libc_free_0_0(v33);
    v12 = v27;
  }
LABEL_16:
  if ( v4 > 0x40 && v31 )
  {
    v28 = v12;
    j_j___libc_free_0_0(v31);
    v12 = v28;
  }
  if ( v12 <= 0 )
    return 256;
  v30 = *(_DWORD *)(a1 + 24);
  if ( v30 > 0x40 )
    sub_C43780((__int64)&v29, (const void **)(a1 + 16));
  else
    v29 = *(_QWORD *)(a1 + 16);
  v14 = *(_DWORD *)(a1 + 8);
  v15 = *(_QWORD *)a1;
  if ( v14 > 0x40 )
    v15 = *(_QWORD *)(v15 + 8LL * ((v14 - 1) >> 6));
  if ( (v15 & (1LL << ((unsigned __int8)v14 - 1))) == 0 )
  {
    v25 = 1LL << ((unsigned __int8)v30 - 1);
    if ( v30 > 0x40 )
      *(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) |= v25;
    else
      v29 |= v25;
  }
  v16 = *(_DWORD *)(a2 + 8);
  v34 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43780((__int64)&v33, (const void **)a2);
    v16 = v34;
    if ( v34 > 0x40 )
    {
      sub_C43D10((__int64)&v33);
      v16 = v34;
      v18 = v33;
      goto LABEL_32;
    }
    v17 = v33;
  }
  else
  {
    v17 = *(_QWORD *)a2;
  }
  v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v17;
  if ( !v16 )
    v18 = 0;
LABEL_32:
  v19 = *(_DWORD *)(a2 + 24);
  v32 = v16;
  v31 = v18;
  v20 = *(_QWORD *)(a2 + 16);
  if ( v19 > 0x40 )
    v20 = *(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6));
  if ( (v20 & (1LL << ((unsigned __int8)v19 - 1))) == 0 )
  {
    v24 = ~(1LL << ((unsigned __int8)v16 - 1));
    if ( v16 <= 0x40 )
    {
      v31 = v18 & v24;
      v21 = sub_C4C880((__int64)&v29, (__int64)&v31);
      goto LABEL_36;
    }
    *(_QWORD *)(v18 + 8LL * ((v16 - 1) >> 6)) &= v24;
    v21 = sub_C4C880((__int64)&v29, (__int64)&v31);
LABEL_41:
    if ( v31 )
      j_j___libc_free_0_0(v31);
    goto LABEL_36;
  }
  v21 = sub_C4C880((__int64)&v29, (__int64)&v31);
  if ( v16 > 0x40 )
    goto LABEL_41;
LABEL_36:
  if ( v30 > 0x40 )
  {
    if ( v29 )
      j_j___libc_free_0_0(v29);
  }
  LOBYTE(result) = 1;
  HIBYTE(result) = v21 > 0;
  return result;
}
