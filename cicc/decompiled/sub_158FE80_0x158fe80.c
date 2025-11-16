// Function: sub_158FE80
// Address: 0x158fe80
//
__int64 __fastcall sub_158FE80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  unsigned int v7; // r13d
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-68h]
  __int64 v12; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-58h]
  __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158A9F0((__int64)&v16, a2);
  sub_158A9F0((__int64)&v14, a3);
  v6 = &v16;
  if ( (int)sub_16A9900(&v14, &v16) < 0 )
    v6 = &v14;
  v11 = *((_DWORD *)v6 + 2);
  if ( v11 > 0x40 )
    sub_16A4FD0(&v10, v6);
  else
    v10 = *v6;
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  v7 = v11;
  if ( v11 <= 0x40 )
  {
    v8 = v10;
    if ( v10 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11) )
      goto LABEL_18;
  }
  else if ( v7 != (unsigned int)sub_16A58F0(&v10) )
  {
    v8 = v10;
LABEL_18:
    v15 = v7;
    v14 = v8;
    v11 = 0;
    sub_16A7490(&v14, 1);
    v9 = v15;
    v15 = 0;
    v17 = v9;
    v16 = v14;
    v13 = *(_DWORD *)(a2 + 8);
    if ( v13 <= 0x40 )
      v12 = 0;
    else
      sub_16A4EF0(&v12, 0, 0);
    sub_15898E0(a1, (__int64)&v12, &v16);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    goto LABEL_31;
  }
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_31:
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return a1;
}
