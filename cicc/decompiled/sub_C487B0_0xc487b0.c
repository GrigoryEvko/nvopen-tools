// Function: sub_C487B0
// Address: 0xc487b0
//
__int64 __fastcall sub_C487B0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // esi
  int v5; // edx
  unsigned int v6; // r14d
  unsigned int v8; // r15d
  unsigned int v9; // eax
  unsigned int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  bool v15; // cc
  unsigned __int64 v16; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  if ( !v4 )
  {
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v5 = a3 % v4;
  v6 = v5;
  if ( !v5 )
  {
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v19 = v4;
  v8 = v4 - v5;
  if ( v4 > 0x40 )
  {
    sub_C43780((__int64)&v18, (const void **)a2);
    v4 = v19;
    if ( v19 > 0x40 )
    {
      sub_C482E0((__int64)&v18, v8);
      goto LABEL_15;
    }
  }
  else
  {
    v18 = *(_QWORD *)a2;
  }
  if ( v8 != v4 )
  {
    v9 = *(_DWORD *)(a2 + 8);
    v18 >>= v8;
    v17 = v9;
    if ( v9 > 0x40 )
      goto LABEL_10;
LABEL_16:
    v16 = *(_QWORD *)a2;
LABEL_17:
    if ( v6 == v9 )
    {
      v12 = 0;
      v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    }
    else
    {
      v12 = v16 << v6;
      v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
      if ( !v9 )
      {
        v14 = 0;
LABEL_20:
        v10 = v19;
        v16 = v14;
        if ( v19 > 0x40 )
          goto LABEL_12;
LABEL_21:
        v11 = v16 | v18;
        v18 |= v16;
        goto LABEL_22;
      }
    }
    v14 = v13 & v12;
    goto LABEL_20;
  }
  v18 = 0;
LABEL_15:
  v9 = *(_DWORD *)(a2 + 8);
  v17 = v9;
  if ( v9 <= 0x40 )
    goto LABEL_16;
LABEL_10:
  sub_C43780((__int64)&v16, (const void **)a2);
  v9 = v17;
  if ( v17 <= 0x40 )
    goto LABEL_17;
  sub_C47690((__int64 *)&v16, v6);
  v10 = v19;
  if ( v19 <= 0x40 )
    goto LABEL_21;
LABEL_12:
  sub_C43BD0(&v18, (__int64 *)&v16);
  v10 = v19;
  v11 = v18;
LABEL_22:
  v15 = v17 <= 0x40;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)a1 = v11;
  v19 = 0;
  if ( !v15 )
  {
    if ( v16 )
    {
      j_j___libc_free_0_0(v16);
      if ( v19 > 0x40 )
      {
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
    }
  }
  return a1;
}
