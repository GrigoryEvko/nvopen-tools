// Function: sub_AB28E0
// Address: 0xab28e0
//
__int64 __fastcall sub_AB28E0(__int64 a1, unsigned int a2, __int64 a3, int a4)
{
  unsigned int v4; // r11d
  unsigned int v5; // r13d
  __int64 v6; // rax
  __int64 *v7; // r15
  unsigned int v8; // r11d
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 *v11; // r9
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 *v14; // rax
  unsigned int v16; // r13d
  unsigned int v17; // eax
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // r11d
  unsigned int v22; // eax
  bool v23; // al
  bool v24; // al
  unsigned int v25; // eax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  unsigned int v29; // eax
  unsigned int v32; // [rsp+10h] [rbp-D0h]
  unsigned int v33; // [rsp+10h] [rbp-D0h]
  unsigned int v34; // [rsp+10h] [rbp-D0h]
  bool v36; // [rsp+18h] [rbp-C8h]
  unsigned int v39; // [rsp+18h] [rbp-C8h]
  unsigned int v40; // [rsp+18h] [rbp-C8h]
  __int64 v41[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD *v42; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-A8h]
  __int64 v44; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-98h]
  unsigned __int64 v46; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v47; // [rsp+58h] [rbp-88h]
  unsigned __int64 v48; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v49; // [rsp+68h] [rbp-78h]
  unsigned __int64 v50; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v51; // [rsp+78h] [rbp-68h]
  __int64 v52[2]; // [rsp+80h] [rbp-60h] BYREF
  unsigned __int64 v53; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v54; // [rsp+98h] [rbp-48h]
  __int64 v55[8]; // [rsp+A0h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a3 + 8);
  if ( a2 == 17 )
  {
    if ( a4 != 1 )
    {
      v14 = sub_9876C0((__int64 *)a3);
      if ( v14 )
      {
        sub_AAE470(a1, v14);
      }
      else
      {
        sub_AB14C0((__int64)&v46, a3);
        sub_AAE470((__int64)&v50, (__int64 *)&v46);
        sub_AB13A0((__int64)&v48, a3);
        sub_AAE470((__int64)&v53, (__int64 *)&v48);
        sub_AB2160(a1, (__int64)&v50, (__int64)&v53, 0);
        sub_969240(v55);
        sub_969240((__int64 *)&v53);
        sub_969240((__int64 *)&v48);
        sub_969240(v52);
        sub_969240((__int64 *)&v50);
        sub_969240((__int64 *)&v46);
      }
      return a1;
    }
    v7 = (__int64 *)&v42;
    sub_AB0910((__int64)&v42, a3);
    v16 = v43;
    if ( v43 <= 0x40 )
    {
      if ( !v42 )
      {
LABEL_80:
        sub_AADB10(a1, v16, 1);
        goto LABEL_37;
      }
      v49 = v43;
      v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v43;
      if ( !v43 )
        v27 = 0;
      v48 = v27;
      sub_C4C950(&v50, &v48, &v42, 0);
      sub_C46A40(&v50, 1);
      v28 = v51;
      v45 = v16;
      v51 = 0;
      v54 = v28;
      v44 = 0;
      v53 = v50;
    }
    else
    {
      if ( v16 - (unsigned int)sub_C444A0(&v42) <= 0x40 && !*v42 )
        goto LABEL_80;
      v49 = v16;
      sub_C43690(&v48, -1, 1);
      sub_C4C950(&v50, &v48, &v42, 0);
      sub_C46A40(&v50, 1);
      v17 = v51;
      v51 = 0;
      v54 = v17;
      v45 = v16;
      v53 = v50;
      sub_C43690(&v44, 0, 0);
    }
    sub_C4C950(&v46, &v44, &v42, 2);
    sub_9875E0(a1, (__int64 *)&v46, (__int64 *)&v53);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    if ( v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    if ( v54 > 0x40 && v53 )
      j_j___libc_free_0_0(v53);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    goto LABEL_37;
  }
  if ( a2 > 0x11 )
  {
    if ( a2 != 25 )
      goto LABEL_84;
    v32 = *(_DWORD *)(a3 + 8);
    sub_9691E0((__int64)&v48, v4, v4, 0, 0);
    v7 = (__int64 *)&v50;
    sub_9691E0((__int64)&v46, v32, 0, 0, 0);
    sub_AADC30((__int64)&v53, (__int64)&v46, (__int64 *)&v48);
    sub_AB2160((__int64)&v50, a3, (__int64)&v53, 0);
    sub_969240(v55);
    sub_969240((__int64 *)&v53);
    sub_969240((__int64 *)&v46);
    sub_969240((__int64 *)&v48);
    if ( sub_AAF7D0((__int64)&v50) )
    {
      sub_AADB10(a1, v32, 1);
      goto LABEL_36;
    }
    sub_AB0910((__int64)v41, (__int64)&v50);
    v8 = v32;
    if ( a4 == 1 )
    {
      sub_9691E0((__int64)&v46, v32, -1, 1u, 0);
      sub_9865C0((__int64)&v48, (__int64)&v46);
      sub_C48380(&v48, v41);
      sub_C46A40(&v48, 1);
      v29 = v49;
      v49 = 0;
      v54 = v29;
      v53 = v48;
      sub_9691E0((__int64)&v44, v32, 0, 0, 0);
      sub_9875E0(a1, &v44, (__int64 *)&v53);
      v11 = &v44;
      goto LABEL_23;
    }
    v47 = v32;
    v9 = ~(1LL << ((unsigned __int8)v32 - 1));
    if ( v32 > 0x40 )
    {
      sub_C43690(&v46, -1, 1);
      v8 = v32;
      v9 = ~(1LL << ((unsigned __int8)v32 - 1));
      if ( v47 > 0x40 )
      {
        *(_QWORD *)(v46 + 8LL * ((v32 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v32 - 1));
        goto LABEL_22;
      }
    }
    else
    {
      v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v32;
      if ( !v32 )
        v10 = 0;
      v46 = v10;
    }
    v46 &= v9;
LABEL_22:
    v33 = v8;
    sub_9865C0((__int64)&v48, (__int64)&v46);
    sub_C44D10(&v48, v41);
    sub_C46A40(&v48, 1);
    v54 = v49;
    v49 = 0;
    v53 = v48;
    sub_986680((__int64)&v42, v33);
    sub_9865C0((__int64)&v44, (__int64)&v42);
    sub_C44D10(&v44, v41);
    sub_9875E0(a1, &v44, (__int64 *)&v53);
    sub_969240(&v44);
    v11 = (__int64 *)&v42;
LABEL_23:
    sub_969240(v11);
    sub_969240((__int64 *)&v53);
    sub_969240((__int64 *)&v48);
    sub_969240((__int64 *)&v46);
    sub_969240(v41);
LABEL_36:
    sub_969240(v52);
LABEL_37:
    sub_969240(v7);
    return a1;
  }
  if ( a2 != 13 )
  {
    if ( a2 == 15 )
    {
      if ( a4 == 1 )
      {
        sub_9691E0((__int64)&v53, v4, 0, 0, 0);
        sub_AB0910((__int64)&v50, a3);
        sub_9875E0(a1, (__int64 *)&v50, (__int64 *)&v53);
        sub_969240((__int64 *)&v50);
        sub_969240((__int64 *)&v53);
        return a1;
      }
      sub_986680((__int64)v41, v4);
      sub_AB14C0((__int64)&v42, a3);
      sub_AB13A0((__int64)&v44, a3);
      v36 = sub_986C60((__int64 *)&v42, v43 - 1);
      if ( v36 )
      {
        sub_9865C0((__int64)&v50, (__int64)v41);
        sub_C45EE0(&v50, &v42);
        v12 = v51;
        v51 = 0;
        v54 = v12;
        v53 = v50;
      }
      else
      {
        sub_9865C0((__int64)&v53, (__int64)v41);
      }
      v13 = 1LL << ((unsigned __int8)v45 - 1);
      if ( v45 > 0x40 )
      {
        if ( (*(_QWORD *)(v44 + 8LL * ((v45 - 1) >> 6)) & v13) != 0 )
          goto LABEL_30;
        v34 = v45;
        v23 = v34 == (unsigned int)sub_C444A0(&v44);
      }
      else
      {
        if ( (v13 & v44) != 0 )
          goto LABEL_30;
        v23 = v44 == 0;
      }
      if ( !v23 )
      {
        sub_9865C0((__int64)&v46, (__int64)v41);
        sub_C45EE0(&v46, &v44);
        goto LABEL_10;
      }
LABEL_30:
      sub_9865C0((__int64)&v48, (__int64)v41);
      sub_9875E0(a1, (__int64 *)&v48, (__int64 *)&v53);
      sub_969240((__int64 *)&v48);
      goto LABEL_11;
    }
LABEL_84:
    BUG();
  }
  if ( a4 != 1 )
  {
    sub_986680((__int64)v41, v4);
    sub_AB14C0((__int64)&v42, a3);
    sub_AB13A0((__int64)&v44, a3);
    v5 = v45;
    v6 = 1LL << ((unsigned __int8)v45 - 1);
    if ( v45 > 0x40 )
    {
      if ( (*(_QWORD *)(v44 + 8LL * ((v45 - 1) >> 6)) & v6) != 0 )
        goto LABEL_7;
      v24 = v5 == (unsigned int)sub_C444A0(&v44);
    }
    else
    {
      if ( (v44 & v6) != 0 )
        goto LABEL_7;
      v24 = v44 == 0;
    }
    if ( !v24 )
    {
      sub_9865C0((__int64)&v50, (__int64)v41);
      sub_C46B40(&v50, &v44);
      v25 = v51;
      v36 = 1;
      v51 = 0;
      v54 = v25;
      v53 = v50;
LABEL_8:
      if ( sub_986C60((__int64 *)&v42, v43 - 1) )
      {
        sub_9865C0((__int64)&v46, (__int64)v41);
        sub_C46B40(&v46, &v42);
LABEL_10:
        v49 = v47;
        v47 = 0;
        v48 = v46;
        sub_9875E0(a1, (__int64 *)&v48, (__int64 *)&v53);
        sub_969240((__int64 *)&v48);
        sub_969240((__int64 *)&v46);
LABEL_11:
        sub_969240((__int64 *)&v53);
        if ( v36 )
          sub_969240((__int64 *)&v50);
        sub_969240(&v44);
        sub_969240((__int64 *)&v42);
        sub_969240(v41);
        return a1;
      }
      goto LABEL_30;
    }
LABEL_7:
    sub_9865C0((__int64)&v53, (__int64)v41);
    v36 = 0;
    goto LABEL_8;
  }
  v39 = *(_DWORD *)(a3 + 8);
  sub_AB0910((__int64)&v50, a3);
  v21 = v39;
  if ( v51 <= 0x40 )
  {
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v51) & ~v50;
    if ( !v51 )
      v26 = 0;
    v50 = v26;
  }
  else
  {
    sub_C43D10(&v50, a3, v51, v19, v20);
    v21 = v39;
  }
  v40 = v21;
  sub_C46250(&v50);
  v22 = v51;
  v51 = 0;
  v54 = v22;
  v53 = v50;
  sub_9691E0((__int64)&v48, v40, 0, 0, 0);
  sub_9875E0(a1, (__int64 *)&v48, (__int64 *)&v53);
  sub_969240((__int64 *)&v48);
  sub_969240((__int64 *)&v53);
  sub_969240((__int64 *)&v50);
  return a1;
}
