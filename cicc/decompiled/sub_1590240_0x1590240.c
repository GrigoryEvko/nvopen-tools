// Function: sub_1590240
// Address: 0x1590240
//
__int64 __fastcall sub_1590240(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rax
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  int v12; // eax
  unsigned __int64 v13; // [rsp+0h] [rbp-A0h]
  unsigned int v14; // [rsp+8h] [rbp-98h]
  unsigned __int64 v15; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-88h]
  _QWORD *v17; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-78h]
  __int64 v19; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-68h]
  __int64 v21; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-58h]
  __int64 v23; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+68h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158A9F0((__int64)&v15, a2);
  sub_158A9F0((__int64)&v17, a3);
  v5 = v16;
  if ( v16 > 0x40 )
  {
    v5 = sub_16A57B0(&v15);
  }
  else if ( v15 )
  {
    _BitScanReverse64(&v6, v15);
    v5 = v16 - 64 + (v6 ^ 0x3F);
  }
  v7 = v5;
  v14 = v18;
  if ( v18 > 0x40 )
  {
    v13 = v5;
    v12 = sub_16A57B0(&v17);
    v7 = v13;
    if ( v14 - v12 > 0x40 )
      goto LABEL_33;
    v8 = (_QWORD *)*v17;
  }
  else
  {
    v8 = v17;
  }
  if ( v7 > (unsigned __int64)v8 )
  {
    sub_158AAD0((__int64)&v19, a2);
    sub_158AAD0((__int64)&v25, a3);
    sub_16A7E20(&v19, &v25);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    sub_16A7E20(&v15, &v17);
    v9 = v16;
    v16 = 0;
    v22 = v9;
    v21 = v15;
    sub_16A7490(&v21, 1);
    v10 = v22;
    v22 = 0;
    v24 = v10;
    v23 = v21;
    v11 = v20;
    v20 = 0;
    v26 = v11;
    v25 = v19;
    sub_15898E0(a1, (__int64)&v25, &v23);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    goto LABEL_26;
  }
LABEL_33:
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_26:
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
