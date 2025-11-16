// Function: sub_27DE490
// Address: 0x27de490
//
void __fastcall sub_27DE490(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        char a8)
{
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // r12d
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  unsigned __int64 v20; // r9
  unsigned int v21; // edx
  __int64 v22; // rdx
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // rax
  unsigned __int64 i; // r15
  unsigned __int64 *v26; // r12
  unsigned int v27; // eax
  __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // r8
  __int64 v31; // rax
  unsigned int *v32; // rdi
  unsigned int *v33; // rsi
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned __int64 v37; // r13
  unsigned int v38; // r15d
  int v39; // r12d
  unsigned __int64 v40; // rcx
  __int64 v41; // rsi
  unsigned int *v42; // rax
  unsigned int *v43; // rcx
  unsigned int *v44; // rax
  unsigned __int64 v45; // rdx
  unsigned int *v46; // rdx
  __int64 v47; // r8
  unsigned int *v48; // r13
  __int64 v49; // rax
  unsigned int v50; // r14d
  unsigned int *v51; // r12
  unsigned __int64 v52; // rax
  int v53; // edx
  __int64 v54; // r13
  __int64 v55; // rax
  char v56; // al
  unsigned __int64 v57; // rax
  unsigned int *v58; // rax
  unsigned __int64 v59; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v60; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v61; // [rsp+28h] [rbp-C8h]
  unsigned int v62; // [rsp+28h] [rbp-C8h]
  int v63; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v65; // [rsp+48h] [rbp-A8h] BYREF
  unsigned int *v66; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+58h] [rbp-98h]
  _BYTE v68[16]; // [rsp+60h] [rbp-90h] BYREF
  unsigned int *v69; // [rsp+70h] [rbp-80h] BYREF
  __int64 v70; // [rsp+78h] [rbp-78h]
  _BYTE v71[16]; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 *v72; // [rsp+90h] [rbp-60h] BYREF
  __int64 v73; // [rsp+98h] [rbp-58h]
  _BYTE v74[80]; // [rsp+A0h] [rbp-50h] BYREF

  if ( !a6 )
    return;
  v65 = sub_FDD860(a6, a3);
  v61 = sub_FDD860(a6, a4);
  v11 = sub_FF0430(a7, a3, a5);
  v60 = sub_1098D20(&v65, v11);
  v12 = v65 - v61;
  if ( v61 >= v65 )
    v12 = 0;
  sub_FE1040(a6, a3, v12);
  v72 = (unsigned __int64 *)v74;
  v73 = 0x400000000LL;
  v13 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3 + 48 == v13 )
    goto LABEL_60;
  if ( !v13 )
    BUG();
  v14 = v13 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA || (v63 = sub_B46E30(v14)) == 0 )
  {
LABEL_60:
    v23 = (unsigned __int64 *)v74;
    v21 = 0;
LABEL_61:
    v57 = *v23;
    v66 = (unsigned int *)v68;
    v67 = 0x400000000LL;
    if ( v57 )
    {
      v32 = (unsigned int *)v68;
      v33 = (unsigned int *)v68;
      goto LABEL_27;
    }
    goto LABEL_36;
  }
  v15 = 0;
  do
  {
    v22 = sub_B46EC0(v14, v15);
    if ( a5 != v22 )
    {
      v16 = sub_FF0430(a7, a3, v22);
      v17 = sub_1098D20(&v65, v16);
      v19 = (unsigned int)v73;
      v20 = (unsigned int)v73 + 1LL;
      if ( v20 <= HIDWORD(v73) )
        goto LABEL_10;
LABEL_15:
      v59 = v17;
      sub_C8D5F0((__int64)&v72, v74, v20, 8u, v18, v20);
      v19 = (unsigned int)v73;
      v17 = v59;
      goto LABEL_10;
    }
    v19 = (unsigned int)v73;
    v17 = v60 - v61;
    v20 = (unsigned int)v73 + 1LL;
    if ( v61 >= v60 )
      v17 = 0;
    if ( v20 > HIDWORD(v73) )
      goto LABEL_15;
LABEL_10:
    ++v15;
    v72[v19] = v17;
    v21 = v73 + 1;
    LODWORD(v73) = v73 + 1;
  }
  while ( v63 != v15 );
  v23 = &v72[v21];
  if ( v72 == v23 )
    goto LABEL_61;
  v24 = v72 + 1;
  for ( i = *v72; v24 != v23; ++v24 )
  {
    if ( i < *v24 )
      i = *v24;
  }
  v66 = (unsigned int *)v68;
  v67 = 0x400000000LL;
  if ( i )
  {
    v26 = v72;
    do
    {
      v27 = sub_F02DD0(*v26, i);
      v29 = (unsigned int)v67;
      v30 = (unsigned int)v67 + 1LL;
      if ( v30 > HIDWORD(v67) )
      {
        v62 = v27;
        sub_C8D5F0((__int64)&v66, v68, (unsigned int)v67 + 1LL, 4u, v30, v28);
        v29 = (unsigned int)v67;
        v27 = v62;
      }
      ++v26;
      v66[v29] = v27;
      v31 = (unsigned int)(v67 + 1);
      LODWORD(v67) = v67 + 1;
    }
    while ( v23 != v26 );
    v32 = v66;
    v33 = &v66[v31];
LABEL_27:
    sub_27DE390(v32, v33);
LABEL_28:
    sub_FF6650(a7, a3, (__int64)&v66);
    goto LABEL_29;
  }
LABEL_36:
  sub_F02DB0(&v69, 1u, v21);
  v37 = (unsigned int)v73;
  v38 = (unsigned int)v69;
  v39 = v73;
  if ( HIDWORD(v67) < (unsigned int)v73 )
  {
    LODWORD(v67) = 0;
    sub_C8D5F0((__int64)&v66, v68, (unsigned int)v73, 4u, v35, v36);
    v58 = v66;
    do
    {
      if ( v58 )
        *v58 = v38;
      ++v58;
      --v37;
    }
    while ( v37 );
    LODWORD(v67) = v39;
    goto LABEL_28;
  }
  v40 = (unsigned int)v67;
  v41 = (unsigned int)v67;
  if ( (unsigned int)v73 <= (unsigned __int64)(unsigned int)v67 )
    v41 = (unsigned int)v73;
  if ( v41 )
  {
    v42 = v66;
    v43 = &v66[v41];
    do
      *v42++ = v38;
    while ( v43 != v42 );
    v40 = (unsigned int)v67;
  }
  if ( v37 > v40 )
  {
    v44 = &v66[v40];
    v45 = v37 - v40;
    if ( v37 != v40 )
    {
      do
      {
        if ( v44 )
          *v44 = v38;
        ++v44;
        --v45;
      }
      while ( v45 );
    }
  }
  LODWORD(v67) = v39;
  sub_FF6650(a7, a3, (__int64)&v66);
LABEL_29:
  if ( (unsigned int)v67 > 1uLL && a8 )
  {
    v70 = 0x400000000LL;
    v46 = (unsigned int *)v71;
    v47 = (__int64)&v66[(unsigned int)v67];
    v69 = (unsigned int *)v71;
    v48 = v66 + 1;
    v49 = 0;
    v50 = *v66;
    v51 = (unsigned int *)v47;
    while ( 1 )
    {
      v46[v49] = v50;
      v49 = (unsigned int)(v70 + 1);
      LODWORD(v70) = v70 + 1;
      if ( v51 == v48 )
        break;
      v50 = *v48;
      if ( v49 + 1 > (unsigned __int64)HIDWORD(v70) )
      {
        sub_C8D5F0((__int64)&v69, v71, v49 + 1, 4u, v47, v34);
        v49 = (unsigned int)v70;
      }
      v46 = v69;
      ++v48;
    }
    v52 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a3 + 48 == v52 )
    {
      v54 = 0;
    }
    else
    {
      if ( !v52 )
        BUG();
      v53 = *(unsigned __int8 *)(v52 - 24);
      v54 = 0;
      v55 = v52 - 24;
      if ( (unsigned int)(v53 - 30) < 0xB )
        v54 = v55;
    }
    v56 = sub_BC87E0(v54);
    sub_BC8EC0(v54, v69, (unsigned int)v70, v56);
    if ( v69 != (unsigned int *)v71 )
      _libc_free((unsigned __int64)v69);
  }
  if ( v66 != (unsigned int *)v68 )
    _libc_free((unsigned __int64)v66);
  if ( v72 != (unsigned __int64 *)v74 )
    _libc_free((unsigned __int64)v72);
}
