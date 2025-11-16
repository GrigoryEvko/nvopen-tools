// Function: sub_1477D10
// Address: 0x1477d10
//
__int64 __fastcall sub_1477D10(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 *v7; // rax
  unsigned int v8; // eax
  __int64 v9; // r12
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-48h]
  unsigned __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]
  unsigned __int64 v15; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-28h]

  v4 = sub_1456040(a1);
  v5 = sub_1456C90(a3, v4);
  *a2 = 36;
  v6 = v5;
  v7 = sub_1477920(a3, a1, 0);
  sub_158A9F0(&v13, v7);
  v12 = v6;
  if ( v6 > 0x40 )
    sub_16A4EF0(&v11, 0, 0);
  else
    v11 = 0;
  if ( v14 > 0x40 )
    sub_16A8F40(&v13);
  else
    v13 = ~v13 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
  sub_16A7400(&v13);
  sub_16A7200(&v13, &v11);
  v8 = v14;
  v14 = 0;
  v16 = v8;
  v15 = v13;
  v9 = sub_145CF40(a3, (__int64)&v15);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return v9;
}
