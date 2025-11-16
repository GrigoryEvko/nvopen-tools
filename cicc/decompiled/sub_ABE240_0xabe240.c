// Function: sub_ABE240
// Address: 0xabe240
//
__int64 __fastcall sub_ABE240(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // edi
  __int64 v8; // rsi
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned int v13; // ebx
  __int64 v15; // r9
  int v16; // eax
  int v17; // eax
  int v18; // ebx
  int v19; // [rsp+14h] [rbp-BCh]
  int v20; // [rsp+14h] [rbp-BCh]
  int v21; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-A8h]
  __int64 v24; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-98h]
  __int64 v26; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-88h]
  __int64 v28; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-78h]
  __int64 v30; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+68h] [rbp-68h]
  unsigned __int64 v32; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+78h] [rbp-58h]
  __int64 v34; // [rsp+80h] [rbp-50h] BYREF
  int v35; // [rsp+88h] [rbp-48h]
  __int64 v36; // [rsp+90h] [rbp-40h] BYREF
  int v37; // [rsp+98h] [rbp-38h]

  if ( sub_AAF7D0(a1) || sub_AAF7D0(a2) )
    return 2;
  sub_AB14C0((__int64)&v22, a1);
  sub_AB13A0((__int64)&v24, a1);
  sub_AB14C0((__int64)&v26, a2);
  sub_AB13A0((__int64)&v28, a2);
  sub_986680((__int64)&v30, *(_DWORD *)(a1 + 8));
  v2 = *(_DWORD *)(a1 + 8);
  v33 = v2;
  if ( v2 <= 0x40 )
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v3 = 0;
    v32 = v3;
    v4 = ~(1LL << ((unsigned __int8)v2 - 1));
    goto LABEL_7;
  }
  sub_C43690(&v32, -1, 1);
  v4 = ~(1LL << ((unsigned __int8)v2 - 1));
  if ( v33 <= 0x40 )
  {
LABEL_7:
    v32 &= v4;
    goto LABEL_8;
  }
  *(_QWORD *)(v32 + 8LL * ((v2 - 1) >> 6)) &= v4;
LABEL_8:
  v5 = v22;
  if ( v23 > 0x40 )
    v5 = *(_QWORD *)(v22 + 8LL * ((v23 - 1) >> 6));
  if ( (v5 & (1LL << ((unsigned __int8)v23 - 1))) != 0 )
    goto LABEL_13;
  v6 = 1LL << ((unsigned __int8)v29 - 1);
  if ( v29 > 0x40 )
  {
    if ( (*(_QWORD *)(v28 + 8LL * ((v29 - 1) >> 6)) & v6) == 0 )
      goto LABEL_13;
LABEL_35:
    sub_9865C0((__int64)&v34, (__int64)&v32);
    sub_C45EE0(&v34, &v28);
    v37 = v35;
    v35 = 0;
    v36 = v34;
    v20 = sub_C4C880(&v22, &v36);
    sub_969240(&v36);
    v13 = 1;
    sub_969240(&v34);
    if ( v20 > 0 )
      goto LABEL_36;
    goto LABEL_13;
  }
  if ( (v28 & v6) != 0 )
    goto LABEL_35;
LABEL_13:
  v7 = v25;
  v8 = v24;
  v9 = v25 - 1;
  v10 = 1LL << ((unsigned __int8)v25 - 1);
  v11 = v24;
  if ( v25 > 0x40 )
    v11 = *(_QWORD *)(v24 + 8LL * (v9 >> 6));
  if ( (v11 & v10) == 0 )
    goto LABEL_16;
  v15 = v26;
  if ( v27 > 0x40 )
    v15 = *(_QWORD *)(v26 + 8LL * ((v27 - 1) >> 6));
  if ( (v15 & (1LL << ((unsigned __int8)v27 - 1))) != 0 )
    goto LABEL_16;
  sub_9865C0((__int64)&v34, (__int64)&v30);
  sub_C45EE0(&v34, &v26);
  v37 = v35;
  v35 = 0;
  v36 = v34;
  v19 = sub_C4C880(&v24, &v36);
  sub_969240(&v36);
  v13 = 0;
  sub_969240(&v34);
  if ( v19 >= 0 )
  {
    v7 = v25;
    v8 = v24;
    v9 = v25 - 1;
    v10 = 1LL << ((unsigned __int8)v25 - 1);
LABEL_16:
    if ( v7 > 0x40 )
      v8 = *(_QWORD *)(v8 + 8LL * (v9 >> 6));
    if ( (v8 & v10) == 0 && sub_986C60(&v26, v27 - 1) )
    {
      sub_9865C0((__int64)&v34, (__int64)&v32);
      sub_C45EE0(&v34, &v26);
      v16 = v35;
      v35 = 0;
      v37 = v16;
      v36 = v34;
      v21 = sub_C4C880(&v24, &v36);
      sub_969240(&v36);
      sub_969240(&v34);
      if ( v21 > 0 )
        goto LABEL_55;
    }
    if ( !sub_986C60(&v22, v23 - 1) )
      goto LABEL_24;
    v12 = v28;
    if ( v29 > 0x40 )
      v12 = *(_QWORD *)(v28 + 8LL * ((v29 - 1) >> 6));
    if ( (v12 & (1LL << ((unsigned __int8)v29 - 1))) != 0 )
      goto LABEL_24;
    sub_9865C0((__int64)&v34, (__int64)&v30);
    sub_C45EE0(&v34, &v28);
    v17 = v35;
    v35 = 0;
    v37 = v17;
    v36 = v34;
    v18 = sub_C4C880(&v22, &v36);
    sub_969240(&v36);
    sub_969240(&v34);
    if ( v18 < 0 )
LABEL_55:
      v13 = 2;
    else
LABEL_24:
      v13 = 3;
  }
LABEL_36:
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return v13;
}
