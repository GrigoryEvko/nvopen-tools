// Function: sub_1370CF0
// Address: 0x1370cf0
//
__int64 __fastcall sub_1370CF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // edx
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned int v10; // ebx
  bool v11; // cc
  __int64 v12; // r13
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-38h]

  v5 = sub_15E44B0(a3);
  if ( !v6 )
  {
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
  v14 = 128;
  sub_16A4EF0(&v13, v5, 0);
  v16 = 128;
  sub_16A4EF0(&v15, a4, 0);
  v8 = *(_QWORD *)(a2 + 8);
  v18 = 128;
  sub_16A4EF0(&v17, *(_QWORD *)(v8 + 16), 0);
  sub_16A7C10(&v13, &v15);
  sub_16A9D70(&v19, &v13, &v17);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  v9 = v19;
  v10 = v20;
  v13 = v19;
  v14 = v20;
  if ( v20 <= 0x40 )
  {
LABEL_10:
    v11 = v18 <= 0x40;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v9;
    if ( !v11 )
      goto LABEL_11;
    goto LABEL_13;
  }
  if ( v10 - (unsigned int)sub_16A57B0(&v13) > 0x40 )
  {
    v9 = -1;
    goto LABEL_10;
  }
  v12 = *(_QWORD *)v9;
  v11 = v18 <= 0x40;
  *(_BYTE *)(a1 + 8) = 1;
  *(_QWORD *)a1 = v12;
  if ( !v11 )
  {
LABEL_11:
    if ( v17 )
      j_j___libc_free_0_0(v17);
  }
LABEL_13:
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return a1;
}
