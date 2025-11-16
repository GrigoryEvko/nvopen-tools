// Function: sub_1484C70
// Address: 0x1484c70
//
__int64 __fastcall sub_1484C70(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        __m128i a7,
        __m128i a8)
{
  __int64 *v11; // rax
  __int64 *v12; // rax
  unsigned __int64 *v13; // rsi
  unsigned int v14; // ecx
  __int64 v15; // r15
  unsigned int v16; // ecx
  __int64 *v17; // rax
  int v18; // eax
  char v19; // dl
  unsigned __int64 *v20; // rsi
  __int64 v21; // r13
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // r12
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  int v28; // eax
  __int64 *v29; // rax
  __int64 *v30; // rax
  char v32; // [rsp+18h] [rbp-B8h]
  __int64 v33; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v35; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-98h]
  unsigned __int64 v37; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-88h]
  unsigned __int64 v39; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-78h]
  unsigned __int64 v41; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-68h]
  unsigned __int64 v43; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-58h]
  unsigned __int64 v45; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v46; // [rsp+88h] [rbp-48h]
  unsigned __int64 v47; // [rsp+90h] [rbp-40h] BYREF
  unsigned int v48; // [rsp+98h] [rbp-38h]

  if ( a6 )
  {
    v11 = sub_1477920((__int64)a1, a2, 1u);
    sub_158ACE0(&v33, v11);
    v12 = sub_1477920((__int64)a1, a3, 1u);
    sub_158ACE0(&v35, v12);
  }
  else
  {
    v29 = sub_1477920((__int64)a1, a2, 0);
    sub_158AAD0(&v33, v29);
    v30 = sub_1477920((__int64)a1, a3, 0);
    sub_158AAD0(&v35, v30);
  }
  v38 = a5;
  if ( a5 > 0x40 )
    sub_16A4EF0(&v37, 1, a6);
  else
    v37 = (0xFFFFFFFFFFFFFFFFLL >> -(char)a5) & 1;
  v13 = &v37;
  if ( (int)sub_16AEA10(&v37, &v35) <= 0 )
    v13 = &v35;
  if ( v36 <= 0x40 && (v14 = *((_DWORD *)v13 + 2), v14 <= 0x40) )
  {
    v26 = *v13;
    v36 = *((_DWORD *)v13 + 2);
    v35 = v26 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
  }
  else
  {
    sub_16A51C0(&v35, v13);
  }
  v40 = a5;
  if ( !a6 )
  {
    if ( a5 <= 0x40 )
      v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a5;
    else
      sub_16A4EF0(&v39, -1, 1);
LABEL_17:
    v46 = v36;
    if ( v36 > 0x40 )
      goto LABEL_14;
    goto LABEL_18;
  }
  v15 = ~(1LL << ((unsigned __int8)a5 - 1));
  if ( a5 <= 0x40 )
  {
    v39 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a5;
    goto LABEL_13;
  }
  sub_16A4EF0(&v39, -1, 1);
  if ( v40 > 0x40 )
  {
    *(_QWORD *)(v39 + 8LL * ((a5 - 1) >> 6)) &= v15;
    goto LABEL_17;
  }
LABEL_13:
  v39 &= v15;
  v46 = v36;
  if ( v36 > 0x40 )
  {
LABEL_14:
    sub_16A4FD0(&v45, &v35);
    goto LABEL_19;
  }
LABEL_18:
  v45 = v35;
LABEL_19:
  sub_16A7800(&v45, 1);
  v16 = v46;
  v46 = 0;
  v48 = v16;
  v47 = v45;
  if ( v16 > 0x40 )
    sub_16A8F40(&v47);
  else
    v47 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v45;
  sub_16A7400(&v47);
  sub_16A7200(&v47, &v39);
  v42 = v48;
  v41 = v47;
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( a6 )
  {
    v17 = sub_1477920((__int64)a1, a4, 1u);
    sub_158ABC0(&v45, v17);
    v18 = sub_16AEA10(&v45, &v41);
    v19 = 0;
    v20 = &v45;
    if ( v18 < 0 )
      goto LABEL_27;
    goto LABEL_26;
  }
  v27 = sub_1477920((__int64)a1, a4, 0);
  sub_158A9F0(&v47, v27);
  v28 = sub_16A9900(&v47, &v41);
  v19 = 1;
  v20 = &v47;
  if ( v28 >= 0 )
  {
LABEL_26:
    v20 = &v41;
LABEL_27:
    v44 = *((_DWORD *)v20 + 2);
    if ( v44 <= 0x40 )
      goto LABEL_28;
    goto LABEL_64;
  }
  v44 = v48;
  if ( v48 <= 0x40 )
  {
LABEL_28:
    v43 = *v20;
    if ( !v19 )
      goto LABEL_29;
LABEL_65:
    sub_135E100((__int64 *)&v47);
    if ( !a6 )
      goto LABEL_30;
    goto LABEL_66;
  }
LABEL_64:
  v32 = v19;
  sub_16A4FD0(&v43, v20);
  if ( v32 )
    goto LABEL_65;
LABEL_29:
  if ( !a6 )
    goto LABEL_30;
LABEL_66:
  sub_135E100((__int64 *)&v45);
LABEL_30:
  v21 = sub_145CF40((__int64)a1, (__int64)&v35);
  v46 = v44;
  if ( v44 > 0x40 )
    sub_16A4FD0(&v45, &v43);
  else
    v45 = v43;
  sub_16A7590(&v45, &v33);
  v22 = v46;
  v46 = 0;
  v48 = v22;
  v47 = v45;
  v23 = sub_145CF40((__int64)a1, (__int64)&v47);
  v24 = sub_1484BE0(a1, v23, v21, 0, a7, a8);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  return v24;
}
