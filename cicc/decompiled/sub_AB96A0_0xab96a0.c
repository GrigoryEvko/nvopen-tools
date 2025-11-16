// Function: sub_AB96A0
// Address: 0xab96a0
//
__int64 __fastcall sub_AB96A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 v6; // [rsp+8h] [rbp-78h]
  unsigned int v7; // [rsp+14h] [rbp-6Ch]
  __int64 v8; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-58h]
  __int64 v10; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v11; // [rsp+38h] [rbp-48h]
  __int64 v12; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+48h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB0910((__int64)&v8, a2);
    sub_AB0A00((__int64)&v10, a3);
    v13 = v9;
    if ( v9 > 0x40 )
      sub_C43780(&v12, &v8);
    else
      v12 = v8;
    sub_C48380(&v12, &v10);
    sub_C46A40(&v12, 1);
    v7 = v13;
    v6 = v12;
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    if ( v9 > 0x40 && v8 )
      j_j___libc_free_0_0(v8);
    sub_AB0A00((__int64)&v10, a2);
    sub_AB0910((__int64)&v12, a3);
    v9 = v11;
    if ( v11 > 0x40 )
      sub_C43780(&v8, &v10);
    else
      v8 = v10;
    sub_C48380(&v8, &v12);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    v13 = v7;
    v12 = v6;
    v5 = v9;
    v9 = 0;
    v11 = v5;
    v10 = v8;
    sub_9875E0(a1, &v10, &v12);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v9 > 0x40 && v8 )
      j_j___libc_free_0_0(v8);
  }
  return a1;
}
