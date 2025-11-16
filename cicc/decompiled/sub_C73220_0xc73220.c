// Function: sub_C73220
// Address: 0xc73220
//
__int64 __fastcall sub_C73220(__int64 a1, __int64 a2, int a3)
{
  unsigned int v4; // r14d
  unsigned int v5; // eax
  unsigned int v6; // r14d
  unsigned int v7; // edi
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  unsigned __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v25; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  if ( v4 == a3 )
  {
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    v23 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v23;
    if ( v23 > 0x40 )
      sub_C43780(a1 + 16, (const void **)(a2 + 16));
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    return a1;
  }
  v5 = *(_DWORD *)(a2 + 24);
  v28 = 1;
  v6 = v4 - a3;
  v27 = 0;
  v30 = 1;
  v29 = 0;
  v26 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43780((__int64)&v25, (const void **)(a2 + 16));
    v5 = v26;
    if ( v26 > 0x40 )
    {
      sub_C47690(&v25, v6);
      v7 = v30;
      goto LABEL_8;
    }
    v7 = v30;
  }
  else
  {
    v7 = 1;
    v25 = *(_QWORD *)(a2 + 16);
  }
  if ( v6 == v5 )
  {
    v8 = 0;
    v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
  }
  else
  {
    v8 = v25 << v6;
    v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
    if ( !v5 )
    {
      v10 = 0;
      goto LABEL_7;
    }
  }
  v10 = v9 & v8;
LABEL_7:
  v25 = v10;
LABEL_8:
  if ( v7 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  v29 = v25;
  v30 = v26;
  v11 = *(_DWORD *)(a2 + 8);
  v26 = v11;
  if ( v11 <= 0x40 )
  {
    v25 = *(_QWORD *)a2;
    goto LABEL_13;
  }
  sub_C43780((__int64)&v25, (const void **)a2);
  v11 = v26;
  if ( v26 <= 0x40 )
  {
LABEL_13:
    if ( v6 == v11 )
    {
      v12 = 0;
      v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    }
    else
    {
      v12 = v25 << v6;
      v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
      if ( !v11 )
      {
        v14 = 0;
        goto LABEL_16;
      }
    }
    v14 = v13 & v12;
LABEL_16:
    v25 = v14;
    goto LABEL_17;
  }
  sub_C47690(&v25, v6);
LABEL_17:
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v27 = v25;
  v15 = v26;
  v28 = v26;
  if ( v30 > 0x40 )
  {
    sub_C44B70((__int64)&v29, v6);
    v15 = v28;
  }
  else
  {
    v16 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v30;
    v17 = 0;
    if ( v30 )
    {
      v18 = v29 << (64 - (unsigned __int8)v30) >> (64 - (unsigned __int8)v30);
      if ( v6 == v30 )
        v17 = v16 & (v18 >> 63);
      else
        v17 = v16 & (v18 >> v6);
    }
    v29 = v17;
  }
  if ( v15 > 0x40 )
  {
    sub_C44B70((__int64)&v27, v6);
    v15 = v28;
    v20 = v27;
  }
  else
  {
    v19 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
    v20 = 0;
    if ( v15 )
    {
      v21 = v27 << (64 - (unsigned __int8)v15) >> (64 - (unsigned __int8)v15);
      if ( v6 == v15 )
        v20 = v19 & (v21 >> 63);
      else
        v20 = v19 & (v21 >> v6);
    }
  }
  *(_DWORD *)(a1 + 8) = v15;
  v22 = v30;
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 24) = v22;
  *(_QWORD *)(a1 + 16) = v29;
  return a1;
}
