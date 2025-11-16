// Function: sub_16A8A10
// Address: 0x16a8a10
//
__int64 __fastcall sub_16A8A10(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // ecx
  int v4; // edx
  unsigned int v6; // r14d
  unsigned int v7; // r15d
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  unsigned int v10; // edx
  unsigned __int64 v11; // rax
  bool v12; // cc
  unsigned __int64 v13; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-48h]
  unsigned __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 8);
  v4 = a3 % v3;
  if ( !v4 )
  {
    *(_DWORD *)(a1 + 8) = v3;
    if ( v3 > 0x40 )
      sub_16A4FD0(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v16 = *(_DWORD *)(a2 + 8);
  v6 = v4;
  v7 = v3 - v4;
  if ( v3 > 0x40 )
  {
    sub_16A4FD0((__int64)&v15, (const void **)a2);
    v3 = v16;
    if ( v16 > 0x40 )
    {
      sub_16A7DC0((__int64 *)&v15, v7);
      v8 = *(_DWORD *)(a2 + 8);
      goto LABEL_10;
    }
    v8 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v15 = *(_QWORD *)a2;
    v8 = v3;
  }
  v9 = 0;
  if ( v7 != v3 )
    v9 = (v15 << v7) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v3);
  v15 = v9;
LABEL_10:
  v14 = v8;
  if ( v8 > 0x40 )
  {
    sub_16A4FD0((__int64)&v13, (const void **)a2);
    v8 = v14;
    if ( v14 > 0x40 )
    {
      sub_16A8110((__int64)&v13, v6);
      goto LABEL_17;
    }
  }
  else
  {
    v13 = *(_QWORD *)a2;
  }
  if ( v6 == v8 )
  {
    v13 = 0;
LABEL_17:
    v10 = v16;
    if ( v16 > 0x40 )
      goto LABEL_14;
LABEL_18:
    v11 = v13 | v15;
    v15 |= v13;
    goto LABEL_19;
  }
  v10 = v16;
  v13 >>= v6;
  if ( v16 <= 0x40 )
    goto LABEL_18;
LABEL_14:
  sub_16A89F0((__int64 *)&v15, (__int64 *)&v13);
  v10 = v16;
  v11 = v15;
LABEL_19:
  v12 = v14 <= 0x40;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)a1 = v11;
  v16 = 0;
  if ( !v12 )
  {
    if ( v13 )
    {
      j_j___libc_free_0_0(v13);
      if ( v16 > 0x40 )
      {
        if ( v15 )
          j_j___libc_free_0_0(v15);
      }
    }
  }
  return a1;
}
