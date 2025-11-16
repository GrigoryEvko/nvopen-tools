// Function: sub_27E6080
// Address: 0x27e6080
//
__int64 __fastcall sub_27E6080(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  char v8; // al
  __int64 *v9; // r14
  unsigned __int64 v10; // rax
  int v11; // edx
  unsigned __int64 v12; // rax
  bool v13; // cf
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  int v16; // edx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  char v25; // cl
  __int64 v26; // rdx
  unsigned __int64 v27; // rbx
  int v28; // eax
  unsigned __int8 *v29; // rbx
  int v30; // eax
  unsigned int v31; // r15d
  int v32; // r14d
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rsi
  _QWORD *v40; // rbx
  _QWORD *v41; // r12
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // eax
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  __int64 v48; // rsi
  unsigned __int64 v49; // [rsp+0h] [rbp-130h]
  unsigned __int64 v50; // [rsp+8h] [rbp-128h]
  __int64 v51; // [rsp+10h] [rbp-120h]
  unsigned int v52; // [rsp+10h] [rbp-120h]
  __int64 v55; // [rsp+28h] [rbp-108h]
  __int64 v56; // [rsp+30h] [rbp-100h]
  _QWORD v57[2]; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v58; // [rsp+58h] [rbp-D8h]
  __int64 v59; // [rsp+60h] [rbp-D0h]
  __int64 **v60; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v61; // [rsp+78h] [rbp-B8h] BYREF
  unsigned __int64 v62; // [rsp+80h] [rbp-B0h] BYREF
  unsigned __int64 v63; // [rsp+88h] [rbp-A8h]
  __int64 i; // [rsp+90h] [rbp-A0h]
  unsigned __int64 v65; // [rsp+98h] [rbp-98h]
  __int64 v66; // [rsp+A0h] [rbp-90h]
  __int64 v67; // [rsp+A8h] [rbp-88h]
  __int64 v68; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD *v69; // [rsp+B8h] [rbp-78h]
  const char *v70; // [rsp+C0h] [rbp-70h]
  unsigned int v71; // [rsp+C8h] [rbp-68h]
  __int16 v72; // [rsp+D0h] [rbp-60h]
  _QWORD *v73; // [rsp+D8h] [rbp-58h]
  unsigned int v74; // [rsp+E8h] [rbp-48h]
  char v75; // [rsp+F0h] [rbp-40h]

  v5 = a1;
  v8 = sub_27DD030(a1, a4);
  v9 = (__int64 *)sub_27DE210(a1, v8);
  v55 = sub_27DE090(a1, v9 != 0);
  v10 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 == a4 + 48 )
  {
    v49 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    v13 = (unsigned int)(v11 - 30) < 0xB;
    v14 = 0;
    if ( v13 )
      v14 = v12;
    v49 = v14;
  }
  v15 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3 + 48 == v15 )
  {
    v50 = 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    v16 = *(unsigned __int8 *)(v15 - 24);
    v17 = v15 - 24;
    v13 = (unsigned int)(v16 - 30) < 0xB;
    v18 = 0;
    if ( v13 )
      v18 = v17;
    v50 = v18;
  }
  v51 = *(_QWORD *)(a3 + 72);
  v68 = (__int64)sub_BD5D20(a3);
  v72 = 773;
  v69 = (_QWORD *)v19;
  v70 = ".thread";
  v56 = sub_AA48A0(a3);
  v20 = sub_22077B0(0x50u);
  v21 = v20;
  if ( v20 )
    sub_AA4D50(v20, v56, (__int64)&v68, v51, a3);
  sub_AA4AF0(v21, a3);
  if ( v9 )
  {
    v52 = sub_FF0430(v55, a2, a3);
    v68 = sub_FDD860(v9, a2);
    v22 = sub_1098D20((unsigned __int64 *)&v68, v52);
    sub_FE1040(v9, v21, v22);
  }
  v68 = 0;
  v71 = 128;
  v23 = (_QWORD *)sub_C7D670(0x2000, 8);
  v70 = 0;
  v69 = v23;
  v61 = 2;
  v24 = &v23[8 * (unsigned __int64)v71];
  v60 = (__int64 **)&unk_49DD7B0;
  v62 = 0;
  v63 = -4096;
  for ( i = 0; v24 != v23; v23 += 8 )
  {
    if ( v23 )
    {
      v25 = v61;
      v23[2] = 0;
      v23[3] = -4096;
      *v23 = &unk_49DD7B0;
      v23[1] = v25 & 6;
      v23[4] = i;
    }
  }
  v26 = *(_QWORD *)(a3 + 56);
  v75 = 0;
  sub_27E2B40(a1, (__int64)&v68, v26, 1, a3 + 48, 0, v21, a2);
  if ( v55 )
    sub_FF5570(v55, a3, v21);
  v27 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v27 == a2 + 48 )
  {
    v29 = 0;
  }
  else
  {
    if ( !v27 )
      BUG();
    v28 = *(unsigned __int8 *)(v27 - 24);
    v29 = (unsigned __int8 *)(v27 - 24);
    if ( (unsigned int)(v28 - 30) >= 0xB )
      v29 = 0;
  }
  v30 = sub_B46E30((__int64)v29);
  if ( v30 )
  {
    v31 = 0;
    v32 = v30;
    do
    {
      while ( a3 != sub_B46EC0((__int64)v29, v31) )
      {
        if ( v32 == ++v31 )
          goto LABEL_30;
      }
      sub_AA5980(a3, a2, 1u);
      sub_B46F90(v29, v31++, v21);
    }
    while ( v32 != v31 );
LABEL_30:
    v5 = a1;
  }
  sub_27DBE10(*(_QWORD *)(v50 - 32), a3, v21, (__int64)&v68);
  sub_27DBE10(*(_QWORD *)(v50 - 64), a3, v21, (__int64)&v68);
  v33 = *(_QWORD *)(v5 + 48);
  v34 = *(_QWORD *)(v49 - 32);
  v60 = (__int64 **)v21;
  v61 = v34 & 0xFFFFFFFFFFFFFFFBLL;
  v35 = *(_QWORD *)(v49 - 64);
  v62 = v21;
  v63 = v35 & 0xFFFFFFFFFFFFFFFBLL;
  i = a2;
  v65 = v21 & 0xFFFFFFFFFFFFFFFBLL;
  v66 = a2;
  v67 = a3 | 4;
  sub_FFDB80(v33, (unsigned __int64 *)&v60, 4, a2, v36, v37);
  sub_27E1050(v5, a3, v21, (__int64)&v68);
  sub_F61E50(v21, *(__int64 **)(v5 + 16));
  sub_F61E50(a3, *(__int64 **)(v5 + 16));
  v60 = (__int64 **)&v62;
  v62 = v21;
  v61 = 0x100000001LL;
  sub_27E59E0((__int64 *)v5, a4, &v60, a5);
  if ( v60 != (__int64 **)&v62 )
    _libc_free((unsigned __int64)v60);
  if ( v75 )
  {
    v45 = v74;
    v75 = 0;
    if ( v74 )
    {
      v46 = v73;
      v47 = &v73[2 * v74];
      do
      {
        if ( *v46 != -4096 && *v46 != -8192 )
        {
          v48 = v46[1];
          if ( v48 )
            sub_B91220((__int64)(v46 + 1), v48);
        }
        v46 += 2;
      }
      while ( v47 != v46 );
      v45 = v74;
    }
    sub_C7D6A0((__int64)v73, 16LL * v45, 8);
  }
  v38 = v71;
  if ( v71 )
  {
    v40 = v69;
    v57[0] = 2;
    v57[1] = 0;
    v41 = &v69[8 * (unsigned __int64)v71];
    v58 = -4096;
    v60 = (__int64 **)&unk_49DD7B0;
    v42 = -4096;
    v59 = 0;
    v61 = 2;
    v62 = 0;
    v63 = -8192;
    i = 0;
    while ( 1 )
    {
      v43 = v40[3];
      if ( v43 != v42 )
      {
        v42 = v63;
        if ( v43 != v63 )
        {
          v44 = v40[7];
          if ( v44 != -4096 && v44 != 0 && v44 != -8192 )
          {
            sub_BD60C0(v40 + 5);
            v43 = v40[3];
          }
          v42 = v43;
        }
      }
      *v40 = &unk_49DB368;
      if ( v42 != -4096 && v42 != 0 && v42 != -8192 )
        sub_BD60C0(v40 + 1);
      v40 += 8;
      if ( v41 == v40 )
        break;
      v42 = v58;
    }
    v60 = (__int64 **)&unk_49DB368;
    if ( v63 != 0 && v63 != -4096 && v63 != -8192 )
      sub_BD60C0(&v61);
    if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
      sub_BD60C0(v57);
    v38 = v71;
  }
  return sub_C7D6A0((__int64)v69, v38 << 6, 8);
}
