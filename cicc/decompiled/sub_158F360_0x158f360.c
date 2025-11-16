// Function: sub_158F360
// Address: 0x158f360
//
__int64 __fastcall sub_158F360(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v7; // rsi
  __int64 *v8; // rsi
  unsigned int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // [rsp+Ch] [rbp-84h]
  unsigned int v14; // [rsp+Ch] [rbp-84h]
  unsigned int v15; // [rsp+Ch] [rbp-84h]
  __int64 v16; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-78h]
  __int64 v18; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-68h]
  __int64 v20; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-58h]
  __int64 v22; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-48h]
  __int64 v24; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+58h] [rbp-38h]

  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return a1;
  }
  sub_158AAD0((__int64)&v24, a3);
  sub_158AAD0((__int64)&v22, a2);
  v7 = &v24;
  if ( (int)sub_16A9900(&v22, &v24) > 0 )
    v7 = &v22;
  v17 = *((_DWORD *)v7 + 2);
  if ( v17 > 0x40 )
    sub_16A4FD0(&v16, v7);
  else
    v16 = *v7;
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  sub_158A9F0((__int64)&v22, a3);
  sub_158A9F0((__int64)&v20, a2);
  v8 = &v20;
  if ( (int)sub_16A9900(&v20, &v22) <= 0 )
    v8 = &v22;
  v25 = *((_DWORD *)v8 + 2);
  if ( v25 > 0x40 )
    sub_16A4FD0(&v24, v8);
  else
    v24 = *v8;
  sub_16A7490(&v24, 1);
  v9 = v25;
  v10 = v24;
  v19 = v25;
  v18 = v24;
  if ( v21 > 0x40 && v20 )
  {
    v13 = v25;
    j_j___libc_free_0_0(v20);
    v9 = v13;
  }
  if ( v23 > 0x40 && v22 )
  {
    v14 = v9;
    j_j___libc_free_0_0(v22);
    v9 = v14;
  }
  if ( v9 <= 0x40 )
  {
    v11 = v16;
    if ( v10 == v16 )
    {
      sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
      goto LABEL_32;
    }
    goto LABEL_28;
  }
  v15 = v9;
  if ( !(unsigned __int8)sub_16A5220(&v18, &v16) )
  {
    v11 = v16;
    v9 = v15;
LABEL_28:
    v25 = v9;
    v24 = v10;
    v23 = v17;
    v22 = v11;
    v17 = 0;
    sub_15898E0(a1, (__int64)&v22, &v24);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    if ( v25 <= 0x40 )
      goto LABEL_32;
    v12 = v24;
    if ( !v24 )
      goto LABEL_32;
    goto LABEL_36;
  }
  sub_15897D0(a1, *(_DWORD *)(a2 + 8), 1);
  v12 = v10;
  if ( v10 )
LABEL_36:
    j_j___libc_free_0_0(v12);
LABEL_32:
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return a1;
}
