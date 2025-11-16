// Function: sub_1C8DFB0
// Address: 0x1c8dfb0
//
_QWORD *__fastcall sub_1C8DFB0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 **v6; // rbx
  __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 **v11; // r15
  __int64 v12; // rdx
  _QWORD *v13; // r15
  _QWORD *v14; // rax
  unsigned int v16; // r13d
  __int64 v17; // rdx
  __int64 *v18; // r12
  unsigned int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // rbx
  unsigned __int64 *v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  char *v27; // rdx
  _QWORD *v28; // r12
  unsigned __int64 *v29; // r15
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // r15
  unsigned __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rsi
  unsigned __int8 *v39; // rsi
  __int64 v40; // rdi
  _QWORD *v41; // rsi
  unsigned int v42; // edi
  _QWORD *v43; // rcx
  unsigned __int64 *v44; // r14
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rsi
  unsigned __int8 *v48; // rsi
  unsigned __int64 *v49; // [rsp+10h] [rbp-F0h]
  unsigned int v51; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v52; // [rsp+28h] [rbp-D8h] BYREF
  const char *v53; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v54; // [rsp+38h] [rbp-C8h]
  const char *v55; // [rsp+40h] [rbp-C0h] BYREF
  char *v56; // [rsp+48h] [rbp-B8h]
  __int16 v57; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v58[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v59; // [rsp+70h] [rbp-90h]
  unsigned __int8 *v60; // [rsp+80h] [rbp-80h] BYREF
  __int64 v61; // [rsp+88h] [rbp-78h]
  unsigned __int64 *v62; // [rsp+90h] [rbp-70h]
  __int64 v63; // [rsp+98h] [rbp-68h]
  __int64 v64; // [rsp+A0h] [rbp-60h]
  int v65; // [rsp+A8h] [rbp-58h]
  __int64 v66; // [rsp+B0h] [rbp-50h]
  __int64 v67; // [rsp+B8h] [rbp-48h]

  v4 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( !v4 )
    BUG();
  v5 = *(_QWORD *)(v4 + 24);
  if ( !v5 )
  {
    v60 = 0;
    v62 = 0;
    v63 = sub_16498A0(0);
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v61 = 0;
    BUG();
  }
  v6 = *(__int64 ***)a2;
  v64 = 0;
  v63 = sub_16498A0(v5 - 24);
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v9 = *(_QWORD *)(v5 + 16);
  v10 = *(unsigned __int8 **)(v5 + 24);
  v60 = 0;
  v61 = v9;
  v62 = (unsigned __int64 *)v5;
  v58[0] = v10;
  if ( v10 )
  {
    sub_1623A60((__int64)v58, (__int64)v10, 2);
    if ( v60 )
      sub_161E7C0((__int64)&v60, (__int64)v60);
    v60 = v58[0];
    if ( v58[0] )
      sub_1623210((__int64)v58, v58[0], (__int64)&v60);
  }
  v11 = (__int64 **)sub_1646BA0(v6[3], 101);
  v53 = sub_1649960(a2);
  v57 = 773;
  v55 = (const char *)&v53;
  v54 = v12;
  v56 = ".param";
  if ( v11 == *(__int64 ***)a2 )
  {
    v13 = (_QWORD *)a2;
  }
  else if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v59 = 257;
    v13 = (_QWORD *)sub_15FDBD0(48, a2, (__int64)v11, (__int64)v58, 0);
    if ( v61 )
    {
      v44 = v62;
      sub_157E9D0(v61 + 40, (__int64)v13);
      v45 = v13[3];
      v46 = *v44;
      v13[4] = v44;
      v46 &= 0xFFFFFFFFFFFFFFF8LL;
      v13[3] = v46 | v45 & 7;
      *(_QWORD *)(v46 + 8) = v13 + 3;
      *v44 = *v44 & 7 | (unsigned __int64)(v13 + 3);
    }
    sub_164B780((__int64)v13, (__int64 *)&v55);
    if ( v60 )
    {
      v52 = v60;
      sub_1623A60((__int64)&v52, (__int64)v60, 2);
      v47 = v13[6];
      if ( v47 )
        sub_161E7C0((__int64)(v13 + 6), v47);
      v48 = v52;
      v13[6] = v52;
      if ( v48 )
        sub_1623210((__int64)&v52, v48, (__int64)(v13 + 6));
    }
  }
  else
  {
    v13 = (_QWORD *)sub_15A46C0(48, (__int64 ***)a2, v11, 0);
  }
  v14 = *(_QWORD **)(a3 + 8);
  if ( *(_QWORD **)(a3 + 16) != v14 )
    goto LABEL_12;
  v41 = &v14[*(unsigned int *)(a3 + 28)];
  v42 = *(_DWORD *)(a3 + 28);
  if ( v14 == v41 )
  {
LABEL_60:
    if ( v42 >= *(_DWORD *)(a3 + 24) )
    {
LABEL_12:
      sub_16CCBA0(a3, (__int64)v13);
      goto LABEL_13;
    }
    *(_DWORD *)(a3 + 28) = v42 + 1;
    *v41 = v13;
    ++*(_QWORD *)a3;
  }
  else
  {
    v43 = 0;
    while ( v13 != (_QWORD *)*v14 )
    {
      if ( *v14 == -2 )
        v43 = v14;
      if ( v41 == ++v14 )
      {
        if ( !v43 )
          goto LABEL_60;
        *v43 = v13;
        --*(_DWORD *)(a3 + 32);
        ++*(_QWORD *)a3;
        break;
      }
    }
  }
LABEL_13:
  if ( a4 )
  {
    v16 = sub_15E0370(a2);
    v53 = sub_1649960(a2);
    v54 = v17;
    v55 = (const char *)&v53;
    v56 = ".copy";
    v57 = 773;
    v18 = v6[3];
    v19 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v61 + 56) + 40LL)) + 4);
    v59 = 257;
    v51 = v19;
    v20 = sub_1648A60(64, 1u);
    v21 = v20;
    if ( v20 )
      sub_15F8BC0((__int64)v20, v18, v51, 0, (__int64)v58, 0);
    if ( v61 )
    {
      v22 = v62;
      sub_157E9D0(v61 + 40, (__int64)v21);
      v23 = v21[3];
      v24 = *v22;
      v21[4] = v22;
      v24 &= 0xFFFFFFFFFFFFFFF8LL;
      v21[3] = v24 | v23 & 7;
      *(_QWORD *)(v24 + 8) = v21 + 3;
      *v22 = *v22 & 7 | (unsigned __int64)(v21 + 3);
    }
    sub_164B780((__int64)v21, (__int64 *)&v55);
    if ( v60 )
    {
      v52 = v60;
      sub_1623A60((__int64)&v52, (__int64)v60, 2);
      v25 = v21[6];
      if ( v25 )
        sub_161E7C0((__int64)(v21 + 6), v25);
      v26 = v52;
      v21[6] = v52;
      if ( v26 )
        sub_1623210((__int64)&v52, v26, (__int64)(v21 + 6));
    }
    sub_15F8A20((__int64)v21, v16);
    v55 = sub_1649960((__int64)v13);
    v56 = v27;
    v58[0] = (unsigned __int8 *)&v55;
    v59 = 773;
    v58[1] = ".copy";
    v28 = sub_1648A60(64, 1u);
    if ( v28 )
      sub_15F9210((__int64)v28, *(_QWORD *)(*v13 + 24LL), (__int64)v13, 0, 0, 0);
    if ( v61 )
    {
      v29 = v62;
      sub_157E9D0(v61 + 40, (__int64)v28);
      v30 = v28[3];
      v31 = *v29;
      v28[4] = v29;
      v31 &= 0xFFFFFFFFFFFFFFF8LL;
      v28[3] = v31 | v30 & 7;
      *(_QWORD *)(v31 + 8) = v28 + 3;
      *v29 = *v29 & 7 | (unsigned __int64)(v28 + 3);
    }
    sub_164B780((__int64)v28, (__int64 *)v58);
    if ( v60 )
    {
      v53 = (const char *)v60;
      sub_1623A60((__int64)&v53, (__int64)v60, 2);
      v32 = v28[6];
      if ( v32 )
        sub_161E7C0((__int64)(v28 + 6), v32);
      v33 = (unsigned __int8 *)v53;
      v28[6] = v53;
      if ( v33 )
        sub_1623210((__int64)&v53, v33, (__int64)(v28 + 6));
    }
    sub_15F8F50((__int64)v28, v16);
    v59 = 257;
    v34 = sub_1648A60(64, 2u);
    v35 = v34;
    if ( v34 )
      sub_15F9650((__int64)v34, (__int64)v28, (__int64)v21, 0, 0);
    if ( v61 )
    {
      v49 = v62;
      sub_157E9D0(v61 + 40, (__int64)v35);
      v36 = *v49;
      v37 = v35[3] & 7LL;
      v35[4] = v49;
      v36 &= 0xFFFFFFFFFFFFFFF8LL;
      v35[3] = v36 | v37;
      *(_QWORD *)(v36 + 8) = v35 + 3;
      *v49 = *v49 & 7 | (unsigned __int64)(v35 + 3);
    }
    sub_164B780((__int64)v35, (__int64 *)v58);
    if ( v60 )
    {
      v55 = (const char *)v60;
      sub_1623A60((__int64)&v55, (__int64)v60, 2);
      v38 = v35[6];
      if ( v38 )
        sub_161E7C0((__int64)(v35 + 6), v38);
      v39 = (unsigned __int8 *)v55;
      v35[6] = v55;
      if ( v39 )
        sub_1623210((__int64)&v55, v39, (__int64)(v35 + 6));
    }
    v40 = (__int64)v35;
    v13 = v21;
    sub_15F9450(v40, v16);
  }
  if ( v60 )
    sub_161E7C0((__int64)&v60, (__int64)v60);
  return v13;
}
