// Function: sub_C3D860
// Address: 0xc3d860
//
__int64 __fastcall sub_C3D860(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 *a5, unsigned int a6)
{
  __int64 *v6; // r14
  _QWORD *v8; // rbx
  char v9; // al
  _QWORD *v10; // rdx
  int v11; // eax
  int v12; // eax
  int v13; // eax
  int v14; // r14d
  int v15; // r14d
  int v16; // eax
  int v17; // r14d
  int v18; // eax
  int v19; // r14d
  int v20; // eax
  int v21; // r14d
  __int64 *v22; // rax
  __int64 *v23; // rdi
  __int64 *v24; // rdi
  int v25; // eax
  int v26; // r14d
  __int64 v27; // rdx
  char v28; // al
  void **v29; // rdi
  void **v30; // r12
  __int64 v31; // rdi
  bool v32; // zf
  void **v33; // rdi
  __int64 v35; // rax
  int v36; // r15d
  int v37; // eax
  int v38; // r15d
  unsigned int v39; // edx
  int v40; // eax
  char v41; // al
  void **v42; // r15
  __int64 v43; // rdi
  int v44; // eax
  __int64 v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // rdx
  int v48; // eax
  __int64 *v49; // rdi
  int v50; // r14d
  int v51; // r14d
  __int64 *v52; // rdi
  int v53; // eax
  int v54; // r14d
  __int64 v55; // rsi
  int v56; // eax
  int v57; // r14d
  __int64 *v58; // rdi
  int v59; // eax
  int v60; // eax
  __int64 v61; // rdi
  int v62; // r14d
  int v63; // r14d
  __int64 *v64; // rdi
  int v65; // eax
  _QWORD *v66; // rdi
  _QWORD *v67; // rdi
  __int64 v68; // rdi
  int v71; // [rsp+10h] [rbp-B0h]
  int v72; // [rsp+10h] [rbp-B0h]
  unsigned int v73; // [rsp+18h] [rbp-A8h]
  int v74; // [rsp+18h] [rbp-A8h]
  int v75; // [rsp+18h] [rbp-A8h]
  _QWORD *v76; // [rsp+18h] [rbp-A8h]
  _QWORD *v77; // [rsp+18h] [rbp-A8h]
  int v79; // [rsp+20h] [rbp-A0h]
  int v80; // [rsp+20h] [rbp-A0h]
  int v81; // [rsp+20h] [rbp-A0h]
  _QWORD *v82; // [rsp+20h] [rbp-A0h]
  _QWORD *v83; // [rsp+20h] [rbp-A0h]
  _QWORD *v84; // [rsp+20h] [rbp-A0h]
  _QWORD *v86; // [rsp+30h] [rbp-90h] BYREF
  __int64 v87; // [rsp+38h] [rbp-88h]
  char v88; // [rsp+44h] [rbp-7Ch]
  _QWORD *v89[4]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v90; // [rsp+70h] [rbp-50h] BYREF
  __int64 v91; // [rsp+78h] [rbp-48h]
  char v92; // [rsp+84h] [rbp-3Ch]

  v6 = a2;
  v8 = sub_C33340();
  if ( (_QWORD *)*a2 == v8 )
    sub_C3C790(&v86, (_QWORD **)a2);
  else
    sub_C33EB0(&v86, a2);
  if ( v86 == v8 )
    v73 = sub_C3D800((__int64 *)&v86, (__int64)a4, a6);
  else
    v73 = sub_C3ADF0((__int64)&v86, (__int64)a4, a6);
  if ( v8 != v86 )
  {
    v9 = v88 & 7;
    if ( (v88 & 7) != 1 )
      goto LABEL_7;
    v30 = *(void ***)(a1 + 8);
    v31 = (__int64)v30;
    if ( v8 != *v30 )
    {
      sub_C33870((__int64)v30, (__int64)&v86);
      v31 = *(_QWORD *)(a1 + 8);
      goto LABEL_46;
    }
    if ( v30 != (void **)&v86 )
    {
      sub_969EE0(*(_QWORD *)(a1 + 8));
      goto LABEL_55;
    }
    goto LABEL_46;
  }
  v9 = *(_BYTE *)(v87 + 20) & 7;
  if ( v9 == 1 )
  {
    v31 = *(_QWORD *)(a1 + 8);
    v30 = (void **)v31;
    if ( *(_QWORD **)v31 == v8 )
    {
      if ( (_QWORD **)v31 != &v86 )
      {
        v84 = *(_QWORD **)(a1 + 8);
        sub_969EE0(v31);
        sub_C3C840(v84, &v86);
        v31 = *(_QWORD *)(a1 + 8);
      }
      goto LABEL_46;
    }
    if ( (_QWORD **)v31 != &v86 )
    {
      sub_C338F0(v31);
LABEL_55:
      if ( v8 == v86 )
        sub_C3C840(v30, &v86);
      else
        sub_C338E0((__int64)v30, (__int64)&v86);
      v31 = *(_QWORD *)(a1 + 8);
    }
LABEL_46:
    v32 = v8 == *(_QWORD **)(v31 + 24);
    v33 = (void **)(v31 + 24);
    if ( v32 )
      sub_C3CEB0(v33, 0);
    else
      sub_C37310((__int64)v33, 0);
    goto LABEL_48;
  }
LABEL_7:
  v10 = (_QWORD *)*a2;
  if ( !v9 )
  {
    if ( v8 == v10 )
      v75 = sub_C3CD00((__int64)a2, (__int64)a4);
    else
      v75 = sub_C37580((__int64)a2, (__int64)a4);
    v35 = *a5;
    if ( v8 == v86 )
    {
      if ( v8 == (_QWORD *)v35 )
      {
        sub_C3C9E0((__int64 *)&v86, a5);
LABEL_63:
        if ( v8 == v86 )
          v36 = sub_C3D800((__int64 *)&v86, (__int64)a3, a6);
        else
          v36 = sub_C3ADF0((__int64)&v86, (__int64)a3, a6);
        if ( v75 == 2 )
        {
          if ( v8 == v86 )
            v60 = sub_C3D800((__int64 *)&v86, (__int64)a4, a6);
          else
            v60 = sub_C3ADF0((__int64)&v86, (__int64)a4, a6);
          v38 = v60 | v36;
          v39 = a6;
          if ( v8 != v86 )
            goto LABEL_69;
        }
        else
        {
          if ( v8 == v86 )
            v37 = sub_C3D800((__int64 *)&v86, (__int64)a2, a6);
          else
            v37 = sub_C3ADF0((__int64)&v86, (__int64)a2, a6);
          a2 = a4;
          v38 = v37 | v36;
          v39 = a6;
          if ( v8 != v86 )
          {
LABEL_69:
            v40 = sub_C3ADF0((__int64)&v86, (__int64)a2, v39);
            goto LABEL_70;
          }
        }
        v40 = sub_C3D800((__int64 *)&v86, (__int64)a2, v39);
LABEL_70:
        v71 = v38 | v40;
        if ( v8 == v86 )
        {
          v41 = *(_BYTE *)(v87 + 20) & 7;
          if ( v41 == 1 )
            goto LABEL_72;
        }
        else
        {
          v41 = v88 & 7;
          if ( (v88 & 7) == 1 )
          {
LABEL_72:
            v42 = *(void ***)(a1 + 8);
LABEL_73:
            sub_C3C870(v42, (void **)&v86);
            v43 = *(_QWORD *)(a1 + 8);
            if ( v8 == *(_QWORD **)(v43 + 24) )
              sub_C3CEB0((void **)(v43 + 24), 0);
            else
              sub_C37310(v43 + 24, 0);
            v73 = v71;
LABEL_48:
            if ( v8 == v86 )
              goto LABEL_43;
LABEL_49:
            sub_C338F0((__int64)&v86);
            return v73;
          }
        }
        v42 = *(void ***)(a1 + 8);
        if ( !v41 )
          goto LABEL_73;
        if ( v8 == *v42 )
        {
          if ( v8 == v86 )
          {
            sub_C3C9E0(*(__int64 **)(a1 + 8), (__int64 *)&v86);
          }
          else if ( v42 != (void **)&v86 )
          {
            sub_969EE0(*(_QWORD *)(a1 + 8));
LABEL_120:
            if ( v8 == v86 )
              sub_C3C790(v42, &v86);
            else
              sub_C33EB0(v42, (__int64 *)&v86);
          }
        }
        else
        {
          if ( v8 != v86 )
          {
            sub_C33E70(*(__int64 **)(a1 + 8), (__int64 *)&v86);
            goto LABEL_88;
          }
          if ( v42 != (void **)&v86 )
          {
            sub_C338F0(*(_QWORD *)(a1 + 8));
            goto LABEL_120;
          }
        }
LABEL_88:
        if ( v8 == (_QWORD *)*a3 )
          sub_C3C790(&v90, (_QWORD **)a3);
        else
          sub_C33EB0(&v90, a3);
        if ( v8 == v90 )
          v44 = sub_C3D800((__int64 *)&v90, (__int64)a5, a6);
        else
          v44 = sub_C3ADF0((__int64)&v90, (__int64)a5, a6);
        v72 = v71 | v44;
        v45 = *(_QWORD *)(a1 + 8);
        v46 = (__int64 *)(v45 + 24);
        if ( v75 == 2 )
        {
          sub_C3CBE0(v46, v6);
          v61 = *(_QWORD *)(a1 + 8);
          if ( v8 == *(_QWORD **)(v61 + 24) )
            v62 = sub_C3D820((__int64 *)(v61 + 24), (__int64)&v86, a6);
          else
            v62 = sub_C3B1F0(v61 + 24, (__int64)&v86, a6);
          v63 = v72 | v62;
          v64 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
          if ( v8 == (_QWORD *)*v64 )
            v65 = sub_C3D800(v64, (__int64)a4, a6);
          else
            v65 = sub_C3ADF0((__int64)v64, (__int64)a4, a6);
          v51 = v65 | v63;
          goto LABEL_101;
        }
        v47 = *a4;
        if ( v8 == *(_QWORD **)(v45 + 24) )
        {
          if ( (_QWORD *)v47 == v8 )
          {
            sub_C3C9E0(v46, a4);
            goto LABEL_96;
          }
          if ( a4 == v46 )
            goto LABEL_156;
          v77 = (_QWORD *)(v45 + 24);
          sub_969EE0((__int64)v46);
          v66 = v77;
        }
        else
        {
          if ( (_QWORD *)v47 != v8 )
          {
            sub_C33E70(v46, a4);
            goto LABEL_96;
          }
          if ( a4 == v46 )
            goto LABEL_97;
          v76 = (_QWORD *)(v45 + 24);
          sub_C338F0((__int64)v46);
          v66 = v76;
        }
        if ( v8 == (_QWORD *)*a4 )
          sub_C3C790(v66, (_QWORD **)a4);
        else
          sub_C33EB0(v66, a4);
LABEL_96:
        v46 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
        if ( v8 != (_QWORD *)*v46 )
        {
LABEL_97:
          v48 = sub_C3B1F0((__int64)v46, (__int64)&v86, a6);
LABEL_98:
          v81 = v72 | v48;
          v49 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
          if ( v8 == (_QWORD *)*v49 )
            v50 = sub_C3D800(v49, (__int64)v6, a6);
          else
            v50 = sub_C3ADF0((__int64)v49, (__int64)v6, a6);
          v51 = v81 | v50;
LABEL_101:
          v52 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
          if ( v8 == (_QWORD *)*v52 )
            v53 = sub_C3D800(v52, (__int64)&v90, a6);
          else
            v53 = sub_C3ADF0((__int64)v52, (__int64)&v90, a6);
          v54 = v53 | v51;
          sub_91D830(&v90);
LABEL_104:
          v73 = v54;
          goto LABEL_48;
        }
LABEL_156:
        v48 = sub_C3D820(v46, (__int64)&v86, a6);
        goto LABEL_98;
      }
      sub_969EE0((__int64)&v86);
    }
    else
    {
      if ( v8 != (_QWORD *)v35 )
      {
        sub_C33E70((__int64 *)&v86, a5);
        goto LABEL_63;
      }
      sub_C338F0((__int64)&v86);
    }
    if ( v8 == (_QWORD *)*a5 )
      sub_C3C790(&v86, (_QWORD **)a5);
    else
      sub_C33EB0(&v86, a5);
    goto LABEL_63;
  }
  if ( v8 == v10 )
    sub_C3C790(v89, (_QWORD **)a2);
  else
    sub_C33EB0(v89, a2);
  if ( v8 == v89[0] )
    v11 = sub_C3D820((__int64 *)v89, (__int64)&v86, a6);
  else
    v11 = sub_C3B1F0((__int64)v89, (__int64)&v86, a6);
  v74 = v73 | v11;
  if ( v8 == v89[0] )
    sub_C3C790(&v90, v89);
  else
    sub_C33EB0(&v90, (__int64 *)v89);
  if ( v8 == v90 )
    v12 = sub_C3D800((__int64 *)&v90, (__int64)a4, a6);
  else
    v12 = sub_C3ADF0((__int64)&v90, (__int64)a4, a6);
  v79 = v74 | v12;
  if ( v8 == v89[0] )
    v13 = sub_C3D800((__int64 *)v89, (__int64)&v86, a6);
  else
    v13 = sub_C3ADF0((__int64)v89, (__int64)&v86, a6);
  v80 = v79 | v13;
  if ( v8 == v89[0] )
    v14 = sub_C3D820((__int64 *)v89, (__int64)a2, a6);
  else
    v14 = sub_C3B1F0((__int64)v89, (__int64)a2, a6);
  v15 = v80 | v14;
  if ( v8 == v89[0] )
    sub_C3CCB0((__int64)v89);
  else
    sub_C34440((unsigned __int8 *)v89);
  if ( v8 == v90 )
    v16 = sub_C3D800((__int64 *)&v90, (__int64)v89, a6);
  else
    v16 = sub_C3ADF0((__int64)&v90, (__int64)v89, a6);
  v17 = v16 | v15;
  if ( v8 == v90 )
    v18 = sub_C3D800((__int64 *)&v90, (__int64)a3, a6);
  else
    v18 = sub_C3ADF0((__int64)&v90, (__int64)a3, a6);
  v19 = v18 | v17;
  if ( v8 == v90 )
    v20 = sub_C3D800((__int64 *)&v90, (__int64)a5, a6);
  else
    v20 = sub_C3ADF0((__int64)&v90, (__int64)a5, a6);
  v21 = v20 | v19;
  if ( v8 == v90 )
  {
    v22 = (__int64 *)v91;
    if ( (*(_BYTE *)(v91 + 20) & 7) != 3 )
    {
LABEL_31:
      v23 = *(__int64 **)(a1 + 8);
      if ( v8 == (_QWORD *)*v23 )
      {
        if ( v8 == v86 )
        {
          sub_C3C9E0(v23, (__int64 *)&v86);
          goto LABEL_34;
        }
        if ( v23 == (__int64 *)&v86 )
        {
          v24 = (__int64 *)&v86;
          goto LABEL_144;
        }
        v83 = *(_QWORD **)(a1 + 8);
        sub_969EE0((__int64)v23);
        v67 = v83;
      }
      else
      {
        if ( v8 != v86 )
        {
          sub_C33E70(v23, (__int64 *)&v86);
          goto LABEL_34;
        }
        if ( v23 == (__int64 *)&v86 )
        {
          v24 = (__int64 *)&v86;
          goto LABEL_35;
        }
        v82 = *(_QWORD **)(a1 + 8);
        sub_C338F0((__int64)v23);
        v67 = v82;
      }
      if ( v8 == v86 )
        sub_C3C790(v67, &v86);
      else
        sub_C33EB0(v67, (__int64 *)&v86);
LABEL_34:
      v24 = *(__int64 **)(a1 + 8);
      if ( v8 != (_QWORD *)*v24 )
      {
LABEL_35:
        v25 = sub_C3ADF0((__int64)v24, (__int64)&v90, a6);
        goto LABEL_36;
      }
LABEL_144:
      v25 = sub_C3D800(v24, (__int64)&v90, a6);
LABEL_36:
      v26 = v25 | v21;
      v27 = *(_QWORD *)(a1 + 8);
      if ( v8 == *(_QWORD **)v27 )
      {
        v28 = *(_BYTE *)(*(_QWORD *)(v27 + 8) + 20LL) & 7;
        if ( v28 != 1 )
          goto LABEL_38;
      }
      else
      {
        v28 = *(_BYTE *)(v27 + 20) & 7;
        if ( v28 != 1 )
        {
LABEL_38:
          v29 = (void **)(v27 + 24);
          if ( !v28 )
          {
LABEL_39:
            if ( v8 == *(_QWORD **)(v27 + 24) )
              sub_C3CEB0(v29, 0);
            else
              sub_C37310((__int64)v29, 0);
            v73 = v26;
            goto LABEL_42;
          }
          sub_C3C870(v29, (void **)&v86);
          v55 = *(_QWORD *)(a1 + 8);
          if ( v8 == *(_QWORD **)(v55 + 24) )
            v56 = sub_C3D820((__int64 *)(v55 + 24), v55, a6);
          else
            v56 = sub_C3B1F0(v55 + 24, v55, a6);
          v57 = v56 | v26;
          v58 = (__int64 *)(*(_QWORD *)(a1 + 8) + 24LL);
          if ( v8 == (_QWORD *)*v58 )
            v59 = sub_C3D800(v58, (__int64)&v90, a6);
          else
            v59 = sub_C3ADF0((__int64)v58, (__int64)&v90, a6);
          v54 = v59 | v57;
          sub_91D830(&v90);
          sub_91D830(v89);
          goto LABEL_104;
        }
      }
      v29 = (void **)(v27 + 24);
      goto LABEL_39;
    }
  }
  else
  {
    v22 = (__int64 *)&v90;
    if ( (v92 & 7) != 3 )
      goto LABEL_31;
  }
  if ( (*((_BYTE *)v22 + 20) & 8) != 0 )
    goto LABEL_31;
  sub_C3C870(*(void ***)(a1 + 8), (void **)&v86);
  v68 = *(_QWORD *)(a1 + 8);
  if ( v8 == *(_QWORD **)(v68 + 24) )
    sub_C3CEB0((void **)(v68 + 24), 0);
  else
    sub_C37310(v68 + 24, 0);
  v73 = 0;
LABEL_42:
  sub_91D830(&v90);
  sub_91D830(v89);
  if ( v8 != v86 )
    goto LABEL_49;
LABEL_43:
  sub_969EE0((__int64)&v86);
  return v73;
}
