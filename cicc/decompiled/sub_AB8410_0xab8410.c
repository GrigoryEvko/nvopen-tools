// Function: sub_AB8410
// Address: 0xab8410
//
__int64 __fastcall sub_AB8410(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned int v6; // eax
  __int64 *v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // [rsp+20h] [rbp-F0h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-E8h]
  __int64 v11; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-D8h]
  __int64 v13[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v14; // [rsp+50h] [rbp-C0h] BYREF
  unsigned int v15; // [rsp+58h] [rbp-B8h]
  __int64 v16[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v17; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+88h] [rbp-88h]
  __int64 v20; // [rsp+90h] [rbp-80h] BYREF
  __int64 v21; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+A8h] [rbp-68h]
  __int64 v23; // [rsp+B0h] [rbp-60h]
  unsigned int v24; // [rsp+B8h] [rbp-58h]
  __int64 v25; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+C8h] [rbp-48h]
  __int64 v27; // [rsp+D0h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+D8h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB0A90((__int64)&v21, a3);
    sub_AB0A90((__int64)&v18, a2);
    sub_C7BCF0(&v21, &v18);
    v5 = v22;
    v22 = 0;
    v26 = v5;
    v25 = v21;
    v6 = v24;
    v24 = 0;
    v28 = v6;
    v27 = v23;
    sub_AAF050((__int64)v16, (__int64)&v25, 0);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    sub_969240(&v20);
    sub_969240(&v18);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    sub_969240(&v21);
    sub_AAFC20((__int64)&v9, a2, a3);
    sub_AB0910((__int64)v13, a2);
    sub_AB0910((__int64)&v11, a3);
    v7 = &v11;
    if ( (int)sub_C49970(&v11, v13) >= 0 )
      v7 = v13;
    v15 = *((_DWORD *)v7 + 2);
    if ( v15 > 0x40 )
      sub_C43780(&v14, v7);
    else
      v14 = *v7;
    sub_C46A40(&v14, 1);
    v8 = v15;
    v15 = 0;
    v19 = v8;
    v18 = v14;
    v22 = v10;
    if ( v10 > 0x40 )
      sub_C43780(&v21, &v9);
    else
      v21 = v9;
    sub_9875E0((__int64)&v25, &v21, &v18);
    sub_969240(&v21);
    sub_969240(&v18);
    sub_969240(&v14);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    sub_969240(v13);
    sub_AB2160(a1, (__int64)v16, (__int64)&v25, 0);
    sub_969240(&v27);
    sub_969240(&v25);
    sub_969240(&v9);
    sub_969240(&v17);
    sub_969240(v16);
  }
  return a1;
}
