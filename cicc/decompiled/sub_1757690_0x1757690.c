// Function: sub_1757690
// Address: 0x1757690
//
void __fastcall sub_1757690(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r13
  unsigned int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // rbx
  __int64 v11; // rbx
  bool v12; // cc
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-38h]

  v4 = (__int64 *)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 8);
  v20 = v6;
  if ( v6 <= 0x40 )
  {
    v7 = *(_QWORD *)a1;
LABEL_3:
    v8 = *(_QWORD *)(a1 + 16) | v7;
LABEL_4:
    v18 = v6;
    v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v6) & ~v8;
    goto LABEL_5;
  }
  sub_16A4FD0((__int64)&v19, (const void **)a1);
  v6 = v20;
  if ( v20 <= 0x40 )
  {
    v7 = v19;
    goto LABEL_3;
  }
  sub_16A89F0(&v19, v4);
  v6 = v20;
  v8 = v19;
  v20 = 0;
  v22 = v6;
  v21 = v19;
  if ( v6 <= 0x40 )
    goto LABEL_4;
  sub_16A8F40(&v21);
  v18 = v22;
  v17 = v21;
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
LABEL_5:
  if ( *(_DWORD *)(a2 + 8) <= 0x40u && *(_DWORD *)(a1 + 24) <= 0x40u )
  {
    v13 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a2 = v13;
    v14 = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a2 + 8) = v14;
    v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v14;
    if ( (unsigned int)v14 <= 0x40 )
    {
      *(_QWORD *)a2 = v15 & v13;
      v9 = *(_DWORD *)(a1 + 24);
      v22 = v9;
      if ( v9 <= 0x40 )
        goto LABEL_9;
      goto LABEL_23;
    }
    v16 = (unsigned int)((unsigned __int64)(v14 + 63) >> 6) - 1;
    *(_QWORD *)(v13 + 8 * v16) &= v15;
  }
  else
  {
    sub_16A51C0(a2, (__int64)v4);
  }
  v9 = *(_DWORD *)(a1 + 24);
  v22 = v9;
  if ( v9 <= 0x40 )
  {
LABEL_9:
    v10 = *(_QWORD *)(a1 + 16);
    goto LABEL_10;
  }
LABEL_23:
  sub_16A4FD0((__int64)&v21, (const void **)v4);
  v9 = v22;
  if ( v22 > 0x40 )
  {
    sub_16A89F0(&v21, (__int64 *)&v17);
    v9 = v22;
    v11 = v21;
    goto LABEL_11;
  }
  v10 = v21;
LABEL_10:
  v11 = v17 | v10;
  v21 = v11;
LABEL_11:
  v12 = *(_DWORD *)(a3 + 8) <= 0x40u;
  v22 = 0;
  if ( v12 || !*(_QWORD *)a3 )
  {
    *(_QWORD *)a3 = v11;
    *(_DWORD *)(a3 + 8) = v9;
    goto LABEL_18;
  }
  j_j___libc_free_0_0(*(_QWORD *)a3);
  v12 = v22 <= 0x40;
  *(_QWORD *)a3 = v11;
  *(_DWORD *)(a3 + 8) = v9;
  if ( v12 || !v21 )
  {
LABEL_18:
    if ( v18 <= 0x40 )
      return;
    goto LABEL_19;
  }
  j_j___libc_free_0_0(v21);
  if ( v18 <= 0x40 )
    return;
LABEL_19:
  if ( v17 )
    j_j___libc_free_0_0(v17);
}
