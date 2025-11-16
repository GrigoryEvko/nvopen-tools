// Function: sub_1590090
// Address: 0x1590090
//
__int64 __fastcall sub_1590090(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rsi
  unsigned int v7; // r15d
  __int64 v8; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-58h]
  __int64 v10; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158AAD0((__int64)&v12, a3);
  sub_158AAD0((__int64)&v10, a2);
  v6 = &v12;
  if ( (int)sub_16A9900(&v10, &v12) > 0 )
    v6 = &v10;
  v9 = *((_DWORD *)v6 + 2);
  if ( v9 > 0x40 )
    sub_16A4FD0(&v8, v6);
  else
    v8 = *v6;
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  v7 = v9;
  if ( v9 <= 0x40 )
  {
    if ( !v8 )
      goto LABEL_17;
LABEL_22:
    v11 = *(_DWORD *)(a2 + 8);
    if ( v11 <= 0x40 )
    {
      v10 = 0;
    }
    else
    {
      sub_16A4EF0(&v10, 0, 0);
      v7 = v9;
    }
    v13 = v7;
    v12 = v8;
    v9 = 0;
    sub_15898E0(a1, (__int64)&v12, &v10);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    goto LABEL_18;
  }
  if ( v7 != (unsigned int)sub_16A57B0(&v8) )
    goto LABEL_22;
LABEL_17:
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
LABEL_18:
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  return a1;
}
