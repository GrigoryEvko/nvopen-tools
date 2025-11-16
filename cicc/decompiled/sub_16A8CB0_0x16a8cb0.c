// Function: sub_16A8CB0
// Address: 0x16a8cb0
//
__int64 __fastcall sub_16A8CB0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // esi
  int v5; // edx
  unsigned int v7; // r14d
  unsigned int v8; // r15d
  unsigned int v9; // ecx
  unsigned int v10; // edx
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  bool v13; // cc
  unsigned __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-48h]
  unsigned __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v5 = a3 % v4;
  if ( !v5 )
  {
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
      sub_16A4FD0(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v17 = v4;
  v7 = v5;
  v8 = v4 - v5;
  if ( v4 > 0x40 )
  {
    sub_16A4FD0((__int64)&v16, (const void **)a2);
    v4 = v17;
    if ( v17 > 0x40 )
    {
      sub_16A8110((__int64)&v16, v8);
      goto LABEL_14;
    }
  }
  else
  {
    v16 = *(_QWORD *)a2;
  }
  if ( v8 != v4 )
  {
    v16 >>= v8;
    v9 = *(_DWORD *)(a2 + 8);
    v15 = v9;
    if ( v9 > 0x40 )
      goto LABEL_9;
LABEL_15:
    v14 = *(_QWORD *)a2;
LABEL_16:
    v12 = 0;
    if ( v7 != v9 )
      v12 = (v14 << v7) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v9);
    v10 = v17;
    v14 = v12;
    if ( v17 > 0x40 )
      goto LABEL_11;
LABEL_19:
    v11 = v14 | v16;
    v16 |= v14;
    goto LABEL_20;
  }
  v16 = 0;
LABEL_14:
  v9 = *(_DWORD *)(a2 + 8);
  v15 = v9;
  if ( v9 <= 0x40 )
    goto LABEL_15;
LABEL_9:
  sub_16A4FD0((__int64)&v14, (const void **)a2);
  v9 = v15;
  if ( v15 <= 0x40 )
    goto LABEL_16;
  sub_16A7DC0((__int64 *)&v14, v7);
  v10 = v17;
  if ( v17 <= 0x40 )
    goto LABEL_19;
LABEL_11:
  sub_16A89F0((__int64 *)&v16, (__int64 *)&v14);
  v10 = v17;
  v11 = v16;
LABEL_20:
  v13 = v15 <= 0x40;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)a1 = v11;
  v17 = 0;
  if ( !v13 )
  {
    if ( v14 )
    {
      j_j___libc_free_0_0(v14);
      if ( v17 > 0x40 )
      {
        if ( v16 )
          j_j___libc_free_0_0(v16);
      }
    }
  }
  return a1;
}
