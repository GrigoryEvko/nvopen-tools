// Function: sub_15904B0
// Address: 0x15904b0
//
__int64 __fastcall sub_15904B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // [rsp+Ch] [rbp-84h]
  __int64 v10; // [rsp+10h] [rbp-80h]
  __int64 v11; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-68h]
  __int64 v13; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+38h] [rbp-58h]
  __int64 v15; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+48h] [rbp-48h]
  __int64 v17; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+58h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158A9F0((__int64)&v13, a2);
  sub_158AAD0((__int64)&v15, a3);
  v18 = v14;
  if ( v14 > 0x40 )
    sub_16A4FD0(&v17, &v13);
  else
    v17 = v13;
  sub_16A81B0(&v17, &v15);
  sub_16A7490(&v17, 1);
  v9 = v18;
  v12 = v18;
  v10 = v17;
  v11 = v17;
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  sub_158AAD0((__int64)&v15, a2);
  sub_158A9F0((__int64)&v17, a3);
  v14 = v16;
  if ( v16 > 0x40 )
    sub_16A4FD0(&v13, &v15);
  else
    v13 = v15;
  sub_16A81B0(&v13, &v17);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  v6 = v14;
  if ( v14 <= 0x40 )
  {
    v7 = v13;
    if ( v10 != v13 )
      goto LABEL_24;
  }
  else if ( !(unsigned __int8)sub_16A5220(&v13, &v11) )
  {
    v7 = v13;
LABEL_24:
    v12 = 0;
    v18 = v9;
    v16 = v6;
    v17 = v10;
    v15 = v7;
    v14 = 0;
    sub_15898E0(a1, (__int64)&v15, &v17);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    if ( v14 <= 0x40 )
      return a1;
    v8 = v13;
    if ( !v13 )
      return a1;
LABEL_40:
    j_j___libc_free_0_0(v8);
    return a1;
  }
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v9 > 0x40 && v10 )
  {
    v8 = v10;
    goto LABEL_40;
  }
  return a1;
}
