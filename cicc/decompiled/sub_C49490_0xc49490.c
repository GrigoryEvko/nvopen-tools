// Function: sub_C49490
// Address: 0xc49490
//
__int64 __fastcall sub_C49490(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // esi
  int v6; // edx
  unsigned int v7; // r14d
  unsigned int v9; // r15d
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  unsigned int v14; // edx
  unsigned __int64 v15; // rax
  bool v16; // cc
  unsigned __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  if ( !v5 )
  {
    *(_DWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v6 = a3 % v5;
  v7 = v6;
  if ( !v6 )
  {
    *(_DWORD *)(a1 + 8) = v5;
    if ( v5 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v20 = v5;
  v9 = v5 - v6;
  if ( v5 <= 0x40 )
  {
    v19 = *(_QWORD *)a2;
    v10 = v5;
    if ( v5 == v9 )
    {
      v11 = 0;
      v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
      goto LABEL_10;
    }
    goto LABEL_8;
  }
  sub_C43780((__int64)&v19, (const void **)a2);
  v5 = v20;
  if ( v20 <= 0x40 )
  {
    v10 = *(_DWORD *)(a2 + 8);
    v11 = 0;
    if ( v9 == v20 )
    {
LABEL_9:
      v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
      if ( !v5 )
      {
        v13 = 0;
        goto LABEL_11;
      }
LABEL_10:
      v13 = v11 & v12;
LABEL_11:
      v19 = v13;
      goto LABEL_12;
    }
LABEL_8:
    v11 = v19 << v9;
    goto LABEL_9;
  }
  sub_C47690((__int64 *)&v19, v9);
  v10 = *(_DWORD *)(a2 + 8);
LABEL_12:
  v18 = v10;
  if ( v10 <= 0x40 )
  {
    v17 = *(_QWORD *)a2;
    goto LABEL_14;
  }
  sub_C43780((__int64)&v17, (const void **)a2);
  v10 = v18;
  if ( v18 <= 0x40 )
  {
LABEL_14:
    if ( v7 == v10 )
      v17 = 0;
    else
      v17 >>= v7;
    v14 = v20;
    if ( v20 <= 0x40 )
      goto LABEL_17;
LABEL_27:
    sub_C43BD0(&v19, (__int64 *)&v17);
    v14 = v20;
    v15 = v19;
    goto LABEL_18;
  }
  sub_C482E0((__int64)&v17, v7);
  v14 = v20;
  if ( v20 > 0x40 )
    goto LABEL_27;
LABEL_17:
  v15 = v17 | v19;
  v19 |= v17;
LABEL_18:
  v16 = v18 <= 0x40;
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v15;
  v20 = 0;
  if ( !v16 )
  {
    if ( v17 )
    {
      j_j___libc_free_0_0(v17);
      if ( v20 > 0x40 )
      {
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
    }
  }
  return a1;
}
