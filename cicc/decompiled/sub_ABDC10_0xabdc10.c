// Function: sub_ABDC10
// Address: 0xabdc10
//
__int64 __fastcall sub_ABDC10(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // r8d
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned int v13; // ebx
  bool v15; // al
  int v16; // eax
  int v17; // eax
  int v18; // ebx
  __int64 v19; // [rsp+0h] [rbp-D0h]
  unsigned int v20; // [rsp+8h] [rbp-C8h]
  unsigned int v21; // [rsp+14h] [rbp-BCh]
  int v22; // [rsp+14h] [rbp-BCh]
  int v23; // [rsp+14h] [rbp-BCh]
  int v24; // [rsp+18h] [rbp-B8h]
  __int64 v25; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-A8h]
  __int64 v27; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-98h]
  __int64 v29; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-88h]
  __int64 v31; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v32; // [rsp+58h] [rbp-78h]
  __int64 v33; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+68h] [rbp-68h]
  unsigned __int64 v35; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+78h] [rbp-58h]
  __int64 v37; // [rsp+80h] [rbp-50h] BYREF
  int v38; // [rsp+88h] [rbp-48h]
  __int64 v39; // [rsp+90h] [rbp-40h] BYREF
  int v40; // [rsp+98h] [rbp-38h]

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) )
    return 2;
  sub_AB14C0((__int64)&v25, a1);
  sub_AB13A0((__int64)&v27, a1);
  sub_AB14C0((__int64)&v29, a2);
  sub_AB13A0((__int64)&v31, a2);
  sub_986680((__int64)&v33, *(_DWORD *)(a1 + 8));
  v2 = *(_DWORD *)(a1 + 8);
  v36 = v2;
  if ( v2 <= 0x40 )
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v3 = 0;
    v35 = v3;
    v4 = ~(1LL << ((unsigned __int8)v2 - 1));
    goto LABEL_7;
  }
  sub_C43690(&v35, -1, 1);
  v4 = ~(1LL << ((unsigned __int8)v2 - 1));
  if ( v36 <= 0x40 )
  {
LABEL_7:
    v35 &= v4;
    goto LABEL_8;
  }
  *(_QWORD *)(v35 + 8LL * ((v2 - 1) >> 6)) &= v4;
LABEL_8:
  v5 = v25;
  if ( v26 > 0x40 )
    v5 = *(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6));
  if ( (v5 & (1LL << ((unsigned __int8)v26 - 1))) != 0 )
    goto LABEL_13;
  v6 = 1LL << ((unsigned __int8)v30 - 1);
  if ( v30 > 0x40 )
  {
    if ( (*(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) & v6) != 0 )
      goto LABEL_13;
LABEL_33:
    sub_9865C0((__int64)&v37, (__int64)&v35);
    sub_C46B40(&v37, &v29);
    v40 = v38;
    v38 = 0;
    v39 = v37;
    v23 = sub_C4C880(&v25, &v39);
    sub_969240(&v39);
    v13 = 1;
    sub_969240(&v37);
    if ( v23 > 0 )
      goto LABEL_34;
    goto LABEL_13;
  }
  if ( (v29 & v6) == 0 )
    goto LABEL_33;
LABEL_13:
  v7 = v28;
  v8 = v27;
  v9 = v28 - 1;
  v10 = v27;
  v11 = 1LL << ((unsigned __int8)v28 - 1);
  if ( v28 > 0x40 )
    v10 = *(_QWORD *)(v27 + 8LL * (v9 >> 6));
  if ( (v10 & v11) == 0 )
    goto LABEL_16;
  v19 = v27;
  v20 = v28;
  v21 = v28 - 1;
  v15 = sub_986C60(&v31, v32 - 1);
  v9 = v21;
  v7 = v20;
  v8 = v19;
  if ( !v15 )
    goto LABEL_16;
  sub_9865C0((__int64)&v37, (__int64)&v33);
  sub_C46B40(&v37, &v31);
  v40 = v38;
  v38 = 0;
  v39 = v37;
  v22 = sub_C4C880(&v27, &v39);
  sub_969240(&v39);
  v13 = 0;
  sub_969240(&v37);
  if ( v22 >= 0 )
  {
    v7 = v28;
    v8 = v27;
    v9 = v28 - 1;
    v11 = 1LL << ((unsigned __int8)v28 - 1);
LABEL_16:
    if ( v7 > 0x40 )
      v8 = *(_QWORD *)(v8 + 8LL * (v9 >> 6));
    if ( (v8 & v11) == 0 )
    {
      v12 = v31;
      if ( v32 > 0x40 )
        v12 = *(_QWORD *)(v31 + 8LL * ((v32 - 1) >> 6));
      if ( (v12 & (1LL << ((unsigned __int8)v32 - 1))) == 0 )
      {
        sub_9865C0((__int64)&v37, (__int64)&v35);
        sub_C46B40(&v37, &v31);
        v16 = v38;
        v38 = 0;
        v40 = v16;
        v39 = v37;
        v24 = sub_C4C880(&v27, &v39);
        sub_969240(&v39);
        sub_969240(&v37);
        if ( v24 > 0 )
          goto LABEL_53;
      }
    }
    if ( !sub_986C60(&v25, v26 - 1) )
      goto LABEL_24;
    if ( !sub_986C60(&v29, v30 - 1) )
      goto LABEL_24;
    sub_9865C0((__int64)&v37, (__int64)&v33);
    sub_C46B40(&v37, &v29);
    v17 = v38;
    v38 = 0;
    v40 = v17;
    v39 = v37;
    v18 = sub_C4C880(&v25, &v39);
    sub_969240(&v39);
    sub_969240(&v37);
    if ( v18 < 0 )
LABEL_53:
      v13 = 2;
    else
LABEL_24:
      v13 = 3;
  }
LABEL_34:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return v13;
}
