// Function: sub_16A8270
// Address: 0x16a8270
//
__int64 __fastcall sub_16A8270(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rdx
  unsigned __int16 v4; // ax
  unsigned __int64 *v6; // rdx
  __int64 *v7; // rax
  __int64 v8; // rcx
  __int64 *v9; // rcx
  _QWORD *v10; // rax
  unsigned int v11; // r15d
  unsigned int v12; // ebx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  const void *v18; // rax
  unsigned int v19; // edx
  unsigned __int64 v20; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-38h]

  v2 = *((_DWORD *)a2 + 2);
  if ( v2 == 32 )
  {
    v18 = *(const void **)a2;
    *(_DWORD *)(a1 + 8) = 32;
    v19 = (byte_42AEA60[BYTE1(v18)] << 16) | (byte_42AEA60[(unsigned __int8)v18] << 24);
    BYTE1(v19) = byte_42AEA60[BYTE2(v18)];
    LOBYTE(v19) = byte_42AEA60[(unsigned int)v18 >> 24];
    *(_QWORD *)a1 = v19;
    return a1;
  }
  if ( v2 <= 0x20 )
  {
    if ( v2 == 8 )
    {
      v17 = *a2;
      *(_DWORD *)(a1 + 8) = 8;
      *(_QWORD *)a1 = byte_42AEA60[v17];
      return a1;
    }
    if ( v2 == 16 )
    {
      v3 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = 16;
      HIBYTE(v4) = byte_42AEA60[(unsigned __int8)v3];
      LOBYTE(v4) = byte_42AEA60[BYTE1(v3)];
      *(_QWORD *)a1 = v4;
      return a1;
    }
    v21 = *((_DWORD *)a2 + 2);
LABEL_12:
    v10 = *(_QWORD **)a2;
    v23 = v2;
    v11 = v2;
    v20 = (unsigned __int64)v10;
    goto LABEL_13;
  }
  if ( v2 == 64 )
  {
    v6 = &v20;
    v20 = *(_QWORD *)a2;
    v7 = (__int64 *)((char *)&v22 + 7);
    do
    {
      v8 = *(unsigned __int8 *)v6;
      v6 = (unsigned __int64 *)((char *)v6 + 1);
      *(_BYTE *)v7 = byte_42AEA60[v8];
      v9 = v7;
      v7 = (__int64 *)((char *)v7 - 1);
    }
    while ( v9 != (__int64 *)&v22 );
    *(_DWORD *)(a1 + 8) = 64;
    *(_QWORD *)a1 = v22;
    return a1;
  }
  v21 = *((_DWORD *)a2 + 2);
  if ( v2 <= 0x40 )
    goto LABEL_12;
  sub_16A4FD0((__int64)&v20, (const void **)a2);
  v2 = *((_DWORD *)a2 + 2);
  v23 = v2;
  if ( v2 > 0x40 )
  {
    sub_16A4EF0((__int64)&v22, 0, 0);
    v2 = *((_DWORD *)a2 + 2);
    v11 = v21;
    v12 = v23;
    goto LABEL_18;
  }
  v11 = v21;
LABEL_13:
  v22 = 0;
  v12 = v2;
LABEL_18:
  if ( v11 <= 0x40 )
  {
    while ( 1 )
    {
      v13 = v20;
      if ( !v20 )
        goto LABEL_35;
LABEL_21:
      if ( v12 > 0x40 )
      {
        sub_16A7DC0((__int64 *)&v22, 1u);
        v12 = v23;
        v11 = v21;
      }
      else
      {
        v14 = 0;
        if ( v12 != 1 )
          v14 = (2 * v22) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
        v22 = v14;
      }
      LOBYTE(v15) = v20;
      if ( v11 > 0x40 )
        v15 = *(_QWORD *)v20;
      v16 = v15 & 1;
      if ( v12 <= 0x40 )
      {
        v22 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v12) & (v22 | v16);
      }
      else
      {
        *(_QWORD *)v22 |= v16;
        v11 = v21;
      }
      --v2;
      if ( v11 > 0x40 )
        break;
      if ( v11 != 1 )
      {
        v20 >>= 1;
LABEL_17:
        v12 = v23;
        goto LABEL_18;
      }
      v20 = 0;
      v12 = v23;
    }
    sub_16A8110((__int64)&v20, 1u);
    v11 = v21;
    goto LABEL_17;
  }
  if ( v11 - (unsigned int)sub_16A57B0((__int64)&v20) > 0x40 )
    goto LABEL_21;
  v13 = *(_QWORD *)v20;
  if ( *(_QWORD *)v20 )
    goto LABEL_21;
LABEL_35:
  if ( v12 > 0x40 )
  {
    sub_16A7DC0((__int64 *)&v22, v2);
    v12 = v23;
    v13 = v22;
    v11 = v21;
  }
  else if ( v12 != v2 )
  {
    v13 = (v22 << v2) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
  }
  *(_DWORD *)(a1 + 8) = v12;
  *(_QWORD *)a1 = v13;
  if ( v11 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return a1;
}
