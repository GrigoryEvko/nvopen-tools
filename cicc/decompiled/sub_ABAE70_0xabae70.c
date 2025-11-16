// Function: sub_ABAE70
// Address: 0xabae70
//
__int64 __fastcall sub_ABAE70(__int64 a1, __int64 a2, __int64 *a3, int a4, unsigned int a5)
{
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // r14d
  int v16; // eax
  __int64 *v17; // rdx
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-C8h]
  __int64 v20[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v21[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v22; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+48h] [rbp-88h]
  __int64 v24; // [rsp+50h] [rbp-80h]
  unsigned int v25; // [rsp+58h] [rbp-78h]
  __int64 v26; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+68h] [rbp-68h]
  __int64 v28[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v29; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-48h]
  __int64 v31; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+98h] [rbp-38h]

  if ( !sub_AAF7D0(a2) && !sub_AAF7D0((__int64)a3) )
  {
    if ( sub_AAF760(a2) && sub_AAF760((__int64)a3) )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
      return a1;
    }
    sub_AB5480((__int64)&v22, a2, a3);
    if ( (a4 & 2) != 0 )
    {
      sub_ABAB70((__int64)&v26, a2, (__int64)a3);
      sub_AB2160((__int64)&v29, (__int64)&v22, (__int64)&v26, a5);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
      v22 = v29;
      v10 = v30;
      v30 = 0;
      v23 = v10;
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      v24 = v31;
      v11 = v32;
      v32 = 0;
      v25 = v11;
      sub_969240(&v31);
      sub_969240(&v29);
      sub_969240(v28);
      sub_969240(&v26);
    }
    if ( (a4 & 1) != 0 )
    {
      sub_ABA9E0((__int64)&v26, a2, (__int64)a3);
      sub_AB2160((__int64)&v29, (__int64)&v22, (__int64)&v26, a5);
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
      v22 = v29;
      v12 = v30;
      v30 = 0;
      v23 = v12;
      if ( v25 > 0x40 && v24 )
        j_j___libc_free_0_0(v24);
      v24 = v31;
      v13 = v32;
      v32 = 0;
      v25 = v13;
      sub_969240(&v31);
      sub_969240(&v29);
      sub_969240(v28);
      sub_969240(&v26);
    }
    if ( a4 != 3 || sub_AB0760((__int64)&v22) )
      goto LABEL_10;
    sub_AB14C0((__int64)&v26, a2);
    if ( v27 > 0x40 )
    {
      v15 = v27 + 1;
      v19 = v26;
      if ( (*(_QWORD *)(v26 + 8LL * ((v27 - 1) >> 6)) & (1LL << ((unsigned __int8)v27 - 1))) != 0 )
      {
        v16 = sub_C44500(&v26);
        v17 = (__int64 *)v19;
        if ( v15 - v16 > 0x40 )
          goto LABEL_36;
      }
      else
      {
        v18 = sub_C444A0(&v26);
        v17 = (__int64 *)v19;
        if ( v15 - v18 > 0x40 )
          goto LABEL_30;
      }
      v14 = *v17;
    }
    else
    {
      if ( !v27 )
        goto LABEL_36;
      v14 = v26 << (64 - (unsigned __int8)v27) >> (64 - (unsigned __int8)v27);
    }
    if ( v14 > 1 )
    {
LABEL_30:
      sub_969240(&v26);
      goto LABEL_31;
    }
LABEL_36:
    sub_AB14C0((__int64)&v29, (__int64)a3);
    if ( !sub_AAD930((__int64)&v29, 1) )
    {
      sub_969240(&v29);
      sub_969240(&v26);
      goto LABEL_10;
    }
    sub_969240(&v29);
    sub_969240(&v26);
LABEL_31:
    sub_986680((__int64)v21, *(_DWORD *)(a2 + 8));
    sub_9691E0((__int64)v20, *(_DWORD *)(a2 + 8), 0, 0, 0);
    sub_9875E0((__int64)&v26, v20, v21);
    sub_AB2160((__int64)&v29, (__int64)&v22, (__int64)&v26, a5);
    sub_AAD5C0(&v22, &v29);
    sub_969240(&v31);
    sub_969240(&v29);
    sub_969240(v28);
    sub_969240(&v26);
    sub_969240(v20);
    sub_969240(v21);
LABEL_10:
    v9 = v23;
    v23 = 0;
    *(_DWORD *)(a1 + 8) = v9;
    *(_QWORD *)a1 = v22;
    *(_DWORD *)(a1 + 24) = v25;
    *(_QWORD *)(a1 + 16) = v24;
    sub_969240(&v22);
    return a1;
  }
  sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  return a1;
}
