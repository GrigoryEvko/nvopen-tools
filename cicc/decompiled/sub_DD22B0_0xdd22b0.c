// Function: sub_DD22B0
// Address: 0xdd22b0
//
_QWORD *__fastcall sub_DD22B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, char a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // eax
  __int64 *v12; // rcx
  const void **v13; // rsi
  __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned int v16; // edx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  int v19; // eax
  char v20; // dl
  unsigned __int64 *v21; // rsi
  __int64 *v22; // rsi
  _QWORD *v23; // rbx
  unsigned int v24; // eax
  _QWORD *v25; // rax
  _QWORD *v26; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // eax
  unsigned __int64 v33; // rax
  int v34; // eax
  __int64 *v35; // rcx
  unsigned __int64 v36; // rdx
  char v37; // [rsp+0h] [rbp-E0h]
  char v39; // [rsp+1Fh] [rbp-C1h]
  char v40; // [rsp+1Fh] [rbp-C1h]
  __int64 v41; // [rsp+20h] [rbp-C0h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v43; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-A8h]
  __int64 v45; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-98h]
  const void *v47; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-88h]
  unsigned __int64 v49; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-78h]
  unsigned __int64 v51; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v52; // [rsp+78h] [rbp-68h]
  unsigned __int64 v53; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v54; // [rsp+88h] [rbp-58h]
  unsigned __int64 v55; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v56; // [rsp+98h] [rbp-48h]
  unsigned __int64 v57; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v58; // [rsp+A8h] [rbp-38h]

  v39 = a6 & (a5 == 1);
  if ( v39 )
  {
    v30 = sub_D95540(a3);
    return sub_DA2C50((__int64)a1, v30, 0, 0);
  }
  if ( a6 )
  {
    if ( (unsigned __int8)sub_DBEC00((__int64)a1, a3) )
      return (_QWORD *)sub_D970F0((__int64)a1);
    v28 = sub_DBB9F0((__int64)a1, a2, 1u, 0);
    sub_AB14C0((__int64)&v41, v28);
    v29 = sub_DBB9F0((__int64)a1, a3, 1u, 0);
    sub_AB14C0((__int64)&v43, v29);
    v46 = a5;
    if ( a5 <= 0x40 )
      goto LABEL_4;
  }
  else
  {
    v9 = sub_DBB9F0((__int64)a1, a2, 0, 0);
    sub_AB0A00((__int64)&v41, v9);
    v10 = sub_DBB9F0((__int64)a1, a3, 0, 0);
    sub_AB0A00((__int64)&v43, v10);
    v46 = a5;
    if ( a5 <= 0x40 )
    {
LABEL_4:
      v45 = 1;
      goto LABEL_5;
    }
  }
  sub_C43690((__int64)&v45, 1, 0);
LABEL_5:
  if ( a6 )
  {
    v11 = sub_C4C880((__int64)&v45, (__int64)&v43);
    v12 = &v45;
    if ( v11 <= 0 )
      v12 = (__int64 *)&v43;
    v13 = (const void **)v12;
    v48 = *((_DWORD *)v12 + 2);
    if ( v48 <= 0x40 )
    {
LABEL_9:
      v47 = *v13;
      goto LABEL_10;
    }
  }
  else
  {
    v34 = sub_C49970((__int64)&v45, &v43);
    v35 = &v45;
    if ( v34 <= 0 )
      v35 = (__int64 *)&v43;
    v13 = (const void **)v35;
    v48 = *((_DWORD *)v35 + 2);
    if ( v48 <= 0x40 )
      goto LABEL_9;
  }
  sub_C43780((__int64)&v47, v13);
LABEL_10:
  v50 = a5;
  if ( a6 )
  {
    v14 = ~(1LL << ((unsigned __int8)a5 - 1));
    if ( a5 <= 0x40 )
    {
      v15 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a5;
      if ( !a5 )
        v15 = 0;
      v49 = v15;
      goto LABEL_15;
    }
    sub_C43690((__int64)&v49, -1, 1);
    if ( v50 <= 0x40 )
    {
LABEL_15:
      v49 &= v14;
      goto LABEL_16;
    }
    *(_QWORD *)(v49 + 8LL * ((a5 - 1) >> 6)) &= v14;
  }
  else
  {
    if ( a5 <= 0x40 )
    {
      v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a5;
      if ( !a5 )
        v33 = 0;
      v49 = v33;
      v58 = v48;
      if ( v48 <= 0x40 )
        goto LABEL_17;
      goto LABEL_85;
    }
    sub_C43690((__int64)&v49, -1, 1);
  }
LABEL_16:
  v58 = v48;
  if ( v48 <= 0x40 )
  {
LABEL_17:
    v57 = (unsigned __int64)v47;
    goto LABEL_18;
  }
LABEL_85:
  sub_C43780((__int64)&v57, &v47);
LABEL_18:
  sub_C46F20((__int64)&v57, 1u);
  v16 = v58;
  v58 = 0;
  v56 = v16;
  v55 = v57;
  if ( v16 > 0x40 )
  {
    sub_C43D10((__int64)&v55);
  }
  else
  {
    v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & ~v57;
    if ( !v16 )
      v17 = 0;
    v55 = v17;
  }
  sub_C46250((__int64)&v55);
  sub_C45EE0((__int64)&v55, (__int64 *)&v49);
  v52 = v56;
  v51 = v55;
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( a6 )
  {
    v18 = sub_DBB9F0((__int64)a1, a4, 1u, 0);
    sub_AB13A0((__int64)&v55, v18);
    v19 = sub_C4C880((__int64)&v55, (__int64)&v51);
    v20 = a6;
    v21 = &v55;
    if ( v19 < 0 )
      goto LABEL_28;
    goto LABEL_27;
  }
  v31 = sub_DBB9F0((__int64)a1, a4, 0, 0);
  sub_AB0910((__int64)&v57, v31);
  v32 = sub_C49970((__int64)&v57, &v51);
  v39 = 1;
  v20 = 0;
  v21 = &v57;
  if ( v32 >= 0 )
  {
LABEL_27:
    v21 = &v51;
LABEL_28:
    v54 = *((_DWORD *)v21 + 2);
    if ( v54 <= 0x40 )
      goto LABEL_29;
    goto LABEL_73;
  }
  v54 = v58;
  if ( v58 <= 0x40 )
  {
LABEL_29:
    v53 = *v21;
    if ( !v39 )
      goto LABEL_30;
LABEL_74:
    if ( v58 > 0x40 && v57 )
    {
      v40 = v20;
      j_j___libc_free_0_0(v57);
      v20 = v40;
    }
    goto LABEL_30;
  }
LABEL_73:
  v37 = v20;
  sub_C43780((__int64)&v53, (const void **)v21);
  v20 = v37;
  if ( v39 )
    goto LABEL_74;
LABEL_30:
  if ( v20 && v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( a6 )
  {
    v22 = (__int64 *)&v53;
    if ( (int)sub_C4C880((__int64)&v53, (__int64)&v41) <= 0 )
      v22 = &v41;
  }
  else
  {
    v22 = (__int64 *)&v53;
    if ( (int)sub_C49970((__int64)&v53, (unsigned __int64 *)&v41) <= 0 )
      v22 = &v41;
  }
  if ( v54 <= 0x40 && *((_DWORD *)v22 + 2) <= 0x40u )
  {
    v36 = *v22;
    v54 = *((_DWORD *)v22 + 2);
    v53 = v36;
  }
  else
  {
    sub_C43990((__int64)&v53, (__int64)v22);
  }
  v23 = sub_DA26C0(a1, (__int64)&v47);
  v58 = v54;
  if ( v54 > 0x40 )
    sub_C43780((__int64)&v57, (const void **)&v53);
  else
    v57 = v53;
  sub_C46B40((__int64)&v57, &v41);
  v24 = v58;
  v58 = 0;
  v56 = v24;
  v55 = v57;
  v25 = sub_DA26C0(a1, (__int64)&v55);
  v26 = sub_DD21F0(a1, (__int64)v25, (__int64)v23);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 )
  {
    if ( v41 )
      j_j___libc_free_0_0(v41);
  }
  return v26;
}
