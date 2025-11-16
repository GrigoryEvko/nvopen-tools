// Function: sub_C48440
// Address: 0xc48440
//
__int64 __fastcall sub_C48440(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v2; // r12d
  unsigned __int64 *v3; // rdx
  __int64 *v4; // rax
  __int64 v5; // rcx
  __int64 *v6; // rcx
  __int64 v7; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  const void *v11; // rdx
  unsigned __int16 v12; // ax
  __int64 v13; // rax
  unsigned int v14; // edx
  _QWORD *v15; // rax
  unsigned int v16; // r14d
  unsigned int v17; // ebx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-38h]

  v2 = *((_DWORD *)a2 + 2);
  if ( v2 == 16 )
  {
    v11 = *(const void **)a2;
    *(_DWORD *)(a1 + 8) = 16;
    HIBYTE(v12) = byte_3F658A0[(unsigned __int8)v11];
    LOBYTE(v12) = byte_3F658A0[BYTE1(v11)];
    *(_QWORD *)a1 = v12;
    return a1;
  }
  if ( v2 <= 0x10 )
  {
    if ( !v2 )
    {
      v10 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = v10;
      return a1;
    }
    if ( v2 == 8 )
    {
      v9 = *a2;
      *(_DWORD *)(a1 + 8) = 8;
      *(_QWORD *)a1 = byte_3F658A0[v9];
      return a1;
    }
    v26 = *((_DWORD *)a2 + 2);
LABEL_16:
    v15 = *(_QWORD **)a2;
    v28 = v2;
    v16 = v2;
    v25 = (unsigned __int64)v15;
LABEL_17:
    v27 = 0;
    v17 = v2;
    goto LABEL_21;
  }
  if ( v2 == 32 )
  {
    v13 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = 32;
    v14 = (byte_3F658A0[BYTE1(v13)] << 16) | (byte_3F658A0[(unsigned __int8)v13] << 24);
    BYTE1(v14) = byte_3F658A0[BYTE2(v13)];
    LOBYTE(v14) = byte_3F658A0[BYTE3(v13)];
    *(_QWORD *)a1 = v14;
    return a1;
  }
  if ( v2 == 64 )
  {
    v3 = &v25;
    v25 = *(_QWORD *)a2;
    v4 = (__int64 *)((char *)&v27 + 7);
    do
    {
      v5 = *(unsigned __int8 *)v3;
      v3 = (unsigned __int64 *)((char *)v3 + 1);
      *(_BYTE *)v4 = byte_3F658A0[v5];
      v6 = v4;
      v4 = (__int64 *)((char *)v4 - 1);
    }
    while ( v6 != (__int64 *)&v27 );
    v7 = v27;
    *(_DWORD *)(a1 + 8) = 64;
    *(_QWORD *)a1 = v7;
    return a1;
  }
  v26 = *((_DWORD *)a2 + 2);
  if ( v2 <= 0x40 )
    goto LABEL_16;
  sub_C43780((__int64)&v25, (const void **)a2);
  v2 = *((_DWORD *)a2 + 2);
  v28 = v2;
  if ( v2 <= 0x40 )
  {
    v16 = v26;
    goto LABEL_17;
  }
  sub_C43690((__int64)&v27, 0, 0);
  v2 = *((_DWORD *)a2 + 2);
  v16 = v26;
  v17 = v28;
LABEL_21:
  if ( v16 <= 0x40 )
  {
    while ( 1 )
    {
      v18 = v25;
      if ( !v25 )
        goto LABEL_36;
LABEL_24:
      if ( v17 > 0x40 )
      {
        sub_C47690((__int64 *)&v27, 1u);
        v17 = v28;
        v16 = v26;
      }
      else
      {
        v19 = 0;
        if ( v17 >= 2 )
          v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & (2 * v27);
        v27 = v19;
      }
      LOBYTE(v20) = v25;
      if ( v16 > 0x40 )
        v20 = *(_QWORD *)v25;
      v21 = v20 & 1;
      if ( v17 <= 0x40 )
      {
        v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & (v27 | v21);
        if ( !v17 )
          v24 = 0;
        v27 = v24;
      }
      else
      {
        *(_QWORD *)v27 |= v21;
        v16 = v26;
      }
      --v2;
      if ( v16 > 0x40 )
        break;
      if ( v16 != 1 )
      {
        v25 >>= 1;
LABEL_20:
        v17 = v28;
        goto LABEL_21;
      }
      v25 = 0;
      v17 = v28;
    }
    sub_C482E0((__int64)&v25, 1u);
    v16 = v26;
    goto LABEL_20;
  }
  if ( v16 - (unsigned int)sub_C444A0((__int64)&v25) > 0x40 )
    goto LABEL_24;
  v18 = *(_QWORD *)v25;
  if ( *(_QWORD *)v25 )
    goto LABEL_24;
LABEL_36:
  if ( v17 > 0x40 )
  {
    sub_C47690((__int64 *)&v27, v2);
    v17 = v28;
    v18 = v27;
    v16 = v26;
  }
  else
  {
    v22 = 0;
    if ( v17 != v2 )
      v22 = v27 << v2;
    v23 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v17) & v22;
    if ( v17 )
      v18 = v23;
  }
  *(_DWORD *)(a1 + 8) = v17;
  *(_QWORD *)a1 = v18;
  if ( v16 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
