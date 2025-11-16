// Function: sub_17579D0
// Address: 0x17579d0
//
__int64 __fastcall sub_17579D0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v3; // r14
  unsigned int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r15d
  __int64 v10; // rbx
  __int64 v11; // rbx
  bool v12; // cc
  unsigned __int64 v13; // rdi
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-38h]

  v3 = (__int64 *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 8);
  v27 = v6;
  if ( v6 <= 0x40 )
  {
    v7 = *(_QWORD *)a1;
LABEL_3:
    v8 = *(_QWORD *)(a1 + 16) | v7;
LABEL_4:
    v25 = v6;
    v24 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v8;
    goto LABEL_5;
  }
  sub_16A4FD0((__int64)&v26, (const void **)a1);
  v6 = v27;
  if ( v27 <= 0x40 )
  {
    v7 = v26;
    goto LABEL_3;
  }
  sub_16A89F0(&v26, v3);
  v6 = v27;
  v8 = v26;
  v27 = 0;
  v29 = v6;
  v28 = v26;
  if ( v6 <= 0x40 )
    goto LABEL_4;
  sub_16A8F40(&v28);
  v25 = v29;
  v24 = v28;
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
LABEL_5:
  if ( *(_DWORD *)(a2 + 8) <= 0x40u && *(_DWORD *)(a1 + 24) <= 0x40u )
  {
    v15 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a2 = v15;
    v16 = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a2 + 8) = v16;
    v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
    if ( (unsigned int)v16 <= 0x40 )
    {
      *(_QWORD *)a2 = v17 & v15;
      v9 = *(_DWORD *)(a1 + 24);
      v29 = v9;
      if ( v9 <= 0x40 )
        goto LABEL_9;
      goto LABEL_25;
    }
    v23 = (unsigned int)((unsigned __int64)(v16 + 63) >> 6) - 1;
    *(_QWORD *)(v15 + 8 * v23) &= v17;
  }
  else
  {
    sub_16A51C0(a2, (__int64)v3);
  }
  v9 = *(_DWORD *)(a1 + 24);
  v29 = v9;
  if ( v9 <= 0x40 )
  {
LABEL_9:
    v10 = *(_QWORD *)(a1 + 16);
    goto LABEL_10;
  }
LABEL_25:
  sub_16A4FD0((__int64)&v28, (const void **)v3);
  v9 = v29;
  if ( v29 > 0x40 )
  {
    sub_16A89F0(&v28, (__int64 *)&v24);
    v9 = v29;
    v11 = v28;
    goto LABEL_11;
  }
  v10 = v28;
LABEL_10:
  v11 = v24 | v10;
  v28 = v11;
LABEL_11:
  v12 = *((_DWORD *)a3 + 2) <= 0x40u;
  v29 = 0;
  if ( v12 || !*a3 )
  {
    *a3 = v11;
    *((_DWORD *)a3 + 2) = v9;
  }
  else
  {
    j_j___libc_free_0_0(*a3);
    v12 = v29 <= 0x40;
    *a3 = v11;
    *((_DWORD *)a3 + 2) = v9;
    if ( !v12 && v28 )
      j_j___libc_free_0_0(v28);
  }
  v13 = v24;
  result = 1LL << ((unsigned __int8)v25 - 1);
  if ( v25 > 0x40 )
  {
    result &= *(_QWORD *)(v24 + 8LL * ((v25 - 1) >> 6));
    if ( !result )
    {
      if ( !v24 )
        return result;
      return j_j___libc_free_0_0(v13);
    }
  }
  else if ( (result & v24) == 0 )
  {
    return result;
  }
  v18 = *(_DWORD *)(a2 + 8);
  v19 = *(_QWORD *)a2;
  v20 = 1LL << ((unsigned __int8)v18 - 1);
  if ( v18 <= 0x40 )
    *(_QWORD *)a2 = v19 | v20;
  else
    *(_QWORD *)(v19 + 8LL * ((v18 - 1) >> 6)) |= v20;
  v21 = *((_DWORD *)a3 + 2);
  v22 = *a3;
  result = ~(1LL << ((unsigned __int8)v21 - 1));
  if ( v21 > 0x40 )
  {
    *(_QWORD *)(v22 + 8LL * ((v21 - 1) >> 6)) &= result;
  }
  else
  {
    result &= v22;
    *a3 = result;
  }
  if ( v25 > 0x40 )
  {
    v13 = v24;
    if ( v24 )
      return j_j___libc_free_0_0(v13);
  }
  return result;
}
