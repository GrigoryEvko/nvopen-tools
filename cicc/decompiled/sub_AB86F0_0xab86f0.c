// Function: sub_AB86F0
// Address: 0xab86f0
//
__int64 __fastcall sub_AB86F0(__int64 a1, __int64 a2, __int64 a3)
{
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // r8
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  __int64 *v10; // rsi
  __int64 v11; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-D8h]
  __int64 v13[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v14[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v15[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v16; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v17; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v18; // [rsp+88h] [rbp-88h]
  __int64 v19; // [rsp+90h] [rbp-80h] BYREF
  __int64 v20; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+A8h] [rbp-68h]
  __int64 v22; // [rsp+B0h] [rbp-60h] BYREF
  int v23; // [rsp+B8h] [rbp-58h]
  __int64 v24; // [rsp+C0h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+C8h] [rbp-48h]
  __int64 v26; // [rsp+D0h] [rbp-40h] BYREF
  int v27; // [rsp+D8h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB0A90((__int64)&v20, a3);
    sub_AB0A90((__int64)&v17, a2);
    sub_C7BD50(&v20, &v17);
    v25 = v21;
    v21 = 0;
    v24 = v20;
    v5 = v23;
    v23 = 0;
    v27 = v5;
    v26 = v22;
    sub_AAF050((__int64)v15, (__int64)&v24, 0);
    sub_969240(&v26);
    sub_969240(&v24);
    sub_969240(&v19);
    sub_969240(&v17);
    sub_969240(&v22);
    sub_969240(&v20);
    sub_AB8340((__int64)&v24, a3);
    sub_AB8340((__int64)&v20, a2);
    sub_AAFC20((__int64)&v17, (__int64)&v20, (__int64)&v24);
    if ( v18 > 0x40 )
    {
      sub_C43D10(&v17, &v20, v18, v6, v7);
    }
    else
    {
      v8 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v18) & ~v17;
      if ( !v18 )
        v8 = 0;
      v17 = v8;
    }
    sub_C46250(&v17);
    v9 = v18;
    v18 = 0;
    v12 = v9;
    v11 = v17;
    sub_969240(&v17);
    sub_969240(&v22);
    sub_969240(&v20);
    sub_969240(&v26);
    sub_969240(&v24);
    v21 = v12;
    if ( v12 > 0x40 )
      sub_C43780(&v20, &v11);
    else
      v20 = v11;
    sub_AB0A00((__int64)v14, a3);
    sub_AB0A00((__int64)v13, a2);
    v10 = v14;
    if ( (int)sub_C49970(v13, v14) > 0 )
      v10 = v13;
    v18 = *((_DWORD *)v10 + 2);
    if ( v18 > 0x40 )
      sub_C43780(&v17, v10);
    else
      v17 = *v10;
    sub_9875E0((__int64)&v24, &v17, &v20);
    sub_969240(&v17);
    sub_969240(v13);
    sub_969240(v14);
    sub_969240(&v20);
    sub_AB2160(a1, (__int64)v15, (__int64)&v24, 0);
    sub_969240(&v26);
    sub_969240(&v24);
    sub_969240(&v11);
    sub_969240(&v16);
    sub_969240(v15);
  }
  return a1;
}
