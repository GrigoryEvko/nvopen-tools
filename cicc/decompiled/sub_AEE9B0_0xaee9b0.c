// Function: sub_AEE9B0
// Address: 0xaee9b0
//
__int64 __fastcall sub_AEE9B0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r12
  __int64 i; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  _QWORD *v16; // rdi
  __int64 *j; // rbx
  __int64 *v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // rax
  const char *v26; // r12
  unsigned int v27; // ecx
  const char **v28; // rax
  const char *v29; // r8
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  unsigned int v36; // edi
  __int64 *v37; // rsi
  __int64 v38; // r9
  _BYTE *v39; // r10
  __int64 v40; // rcx
  __int64 v41; // r8
  unsigned int v42; // edi
  _QWORD *v43; // rsi
  _BYTE *v44; // r9
  _BYTE *v45; // r12
  int v46; // eax
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 **v51; // rsi
  _QWORD *v52; // rbx
  unsigned int v53; // r14d
  int v54; // r12d
  _BYTE *v55; // r8
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  _BYTE *v59; // r15
  __int64 v60; // rax
  unsigned int v61; // ecx
  const char *v62; // rdx
  __int64 v63; // r8
  _BYTE *v64; // rax
  int v65; // esi
  int v66; // esi
  int v67; // r12d
  int v68; // r11d
  int v69; // edx
  __int64 *v70; // rdi
  unsigned int v71; // r12d
  __int64 *v73; // r12
  const char **v74; // r14
  int v75; // eax
  int v76; // r9d
  int v77; // r9d
  _QWORD *v78; // [rsp+18h] [rbp-148h]
  __int64 *v79; // [rsp+20h] [rbp-140h]
  __int64 *v80; // [rsp+28h] [rbp-138h]
  __int64 *v81; // [rsp+30h] [rbp-130h]
  int v83; // [rsp+48h] [rbp-118h]
  __int64 v84; // [rsp+58h] [rbp-108h]
  _BYTE *v85; // [rsp+58h] [rbp-108h]
  int v86; // [rsp+58h] [rbp-108h]
  __int64 *v87; // [rsp+60h] [rbp-100h]
  __int64 *v88; // [rsp+60h] [rbp-100h]
  int v89; // [rsp+60h] [rbp-100h]
  _BYTE *v90; // [rsp+68h] [rbp-F8h]
  unsigned __int8 v91; // [rsp+77h] [rbp-E9h] BYREF
  __int64 **v92; // [rsp+78h] [rbp-E8h] BYREF
  __int64 *v93; // [rsp+80h] [rbp-E0h] BYREF
  unsigned __int8 *v94; // [rsp+88h] [rbp-D8h]
  __int64 v95; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v96; // [rsp+98h] [rbp-C8h]
  __int64 v97; // [rsp+A0h] [rbp-C0h]
  __int64 v98; // [rsp+A8h] [rbp-B8h]
  __int64 v99; // [rsp+B0h] [rbp-B0h]
  __int64 v100; // [rsp+B8h] [rbp-A8h]
  __int64 v101; // [rsp+C0h] [rbp-A0h]
  __int64 v102; // [rsp+C8h] [rbp-98h]
  unsigned int v103; // [rsp+D0h] [rbp-90h]
  __int64 *v104; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v105; // [rsp+E8h] [rbp-78h]
  _BYTE v106[112]; // [rsp+F0h] [rbp-70h] BYREF

  v91 = 0;
  v1 = sub_BA8CB0(a1, "llvm.dbg.declare", 16);
  if ( v1 )
  {
    v4 = v1;
    for ( i = *(_QWORD *)(v1 + 16); i; i = *(_QWORD *)(v4 + 16) )
      sub_B43D60(*(_QWORD *)(i + 24), "llvm.dbg.declare", v2, v3);
    sub_B2E860(v4);
    v91 = 1;
  }
  v8 = sub_BA8CB0(a1, "llvm.dbg.label", 14);
  if ( v8 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v8 + 16);
      if ( !v9 )
        break;
      sub_B43D60(*(_QWORD *)(v9 + 24), "llvm.dbg.label", v6, v7);
    }
    sub_B2E860(v8);
    v91 = 1;
  }
  v12 = sub_BA8CB0(a1, "llvm.dbg.value", 14);
  if ( v12 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v12 + 16);
      if ( !v13 )
        break;
      sub_B43D60(*(_QWORD *)(v13 + 24), "llvm.dbg.value", v10, v11);
    }
    sub_B2E860(v12);
    v91 = 1;
  }
  v14 = (_QWORD *)a1[10];
  v78 = a1 + 9;
  v15 = a1 + 9;
  while ( v15 != v14 )
  {
    v16 = v14;
    v14 = (_QWORD *)v14[1];
    sub_B91B20(v16);
  }
  for ( j = (__int64 *)a1[2]; a1 + 1 != j; j = (__int64 *)j[1] )
  {
    v18 = j - 7;
    if ( !j )
      v18 = 0;
    sub_B98000(v18, 0);
  }
  v95 = 0;
  v19 = *a1;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v20 = sub_B9C770(v19, 0, 0, 0, 1);
  v21 = 0;
  v22 = sub_B07260(v19, 0, 0, v20, 0, 1);
  v100 = 0;
  v99 = v22;
  v93 = &v95;
  v94 = &v91;
  v23 = (__int64 *)a1[4];
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v79 = v23;
  if ( a1 + 3 != v23 )
  {
    while ( 1 )
    {
      v24 = v79 - 7;
      if ( !v79 )
        v24 = 0;
      v25 = sub_B92180(v24);
      v26 = (const char *)v25;
      if ( v25 )
        break;
LABEL_31:
      v81 = (__int64 *)v24[10];
      v80 = v24 + 9;
      if ( v24 + 9 != v81 )
      {
        while ( 1 )
        {
          if ( !v81 )
            BUG();
          v30 = v81[4];
          if ( v81 + 3 != (__int64 *)v30 )
            break;
LABEL_64:
          v81 = (__int64 *)v81[1];
          if ( v80 == v81 )
            goto LABEL_65;
        }
        while ( 2 )
        {
          if ( !v30 )
          {
            v104 = (__int64 *)&v93;
            v105 = (__int64)a1;
            BUG();
          }
          v104 = (__int64 *)&v93;
          v105 = (__int64)a1;
          if ( *(_QWORD *)(v30 + 24) )
          {
            v31 = sub_B10D00(v30 + 24);
            v32 = sub_B10D40(v30 + 24);
            v33 = v104;
            if ( v31 )
            {
              v84 = v32;
              v87 = v104;
              sub_AEE3F0(*v104, v31);
              v32 = v84;
              v34 = *(unsigned int *)(*v87 + 24);
              if ( (_DWORD)v34 )
              {
                v35 = *(_QWORD *)(*v87 + 8);
                v36 = (v34 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                v37 = (__int64 *)(v35 + 16LL * v36);
                v38 = *v37;
                if ( v31 == *v37 )
                {
LABEL_41:
                  if ( v37 != (__int64 *)(v35 + 16 * v34) )
                  {
                    v39 = (_BYTE *)v37[1];
                    if ( v39 )
                      goto LABEL_43;
                    goto LABEL_44;
                  }
                }
                else
                {
                  v66 = 1;
                  while ( v38 != -4096 )
                  {
                    v68 = v66 + 1;
                    v36 = (v34 - 1) & (v66 + v36);
                    v37 = (__int64 *)(v35 + 16LL * v36);
                    v38 = *v37;
                    if ( v31 == *v37 )
                      goto LABEL_41;
                    v66 = v68;
                  }
                }
              }
              v39 = (_BYTE *)v31;
LABEL_43:
              if ( (unsigned __int8)(*v39 - 5) > 0x1Fu )
LABEL_44:
                v39 = 0;
              *(_BYTE *)v87[1] |= v31 != (_QWORD)v39;
              v33 = v104;
            }
            else
            {
              LODWORD(v39) = 0;
            }
            if ( v32 )
            {
              v83 = (int)v39;
              v85 = (_BYTE *)v32;
              v88 = v33;
              sub_AEE3F0(*v33, v32);
              LODWORD(v39) = v83;
              v40 = *(unsigned int *)(*v88 + 24);
              if ( (_DWORD)v40 )
              {
                v41 = *(_QWORD *)(*v88 + 8);
                v42 = (v40 - 1) & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                v43 = (_QWORD *)(v41 + 16LL * v42);
                v44 = (_BYTE *)*v43;
                if ( (_BYTE *)*v43 == v85 )
                {
LABEL_49:
                  if ( v43 != (_QWORD *)(v41 + 16 * v40) )
                  {
                    v45 = (_BYTE *)v43[1];
                    if ( v45 )
                      goto LABEL_51;
                    goto LABEL_52;
                  }
                }
                else
                {
                  v65 = 1;
                  while ( v44 != (_BYTE *)-4096LL )
                  {
                    v67 = v65 + 1;
                    v42 = (v40 - 1) & (v65 + v42);
                    v43 = (_QWORD *)(v41 + 16LL * v42);
                    v44 = (_BYTE *)*v43;
                    if ( v85 == (_BYTE *)*v43 )
                      goto LABEL_49;
                    v65 = v67;
                  }
                }
              }
              v45 = v85;
LABEL_51:
              if ( (unsigned __int8)(*v45 - 5) > 0x1Fu )
LABEL_52:
                v45 = 0;
              *(_BYTE *)v88[1] |= v85 != v45;
            }
            else
            {
              LODWORD(v45) = 0;
            }
            v86 = (int)v39;
            v89 = sub_B10CF0(v30 + 24);
            v46 = sub_B10CE0(v30 + 24);
            v47 = sub_B01860(*(_QWORD *)v105, v46, v89, v86, (_DWORD)v45, 0, 0, 1);
            sub_B10CB0(&v92, v47);
            if ( (__int64 ***)(v30 + 24) == &v92 )
            {
              if ( v92 )
                sub_B91220(&v92);
            }
            else
            {
              if ( *(_QWORD *)(v30 + 24) )
                sub_B91220(v30 + 24);
              v51 = v92;
              *(_QWORD *)(v30 + 24) = v92;
              if ( v51 )
                sub_B976B0(&v92, v51, v30 + 24, v48, v49, v50);
            }
          }
          v21 = (unsigned __int64)sub_AEF370;
          v92 = &v104;
          sub_AE8EA0(v30 - 24, (__int64 (__fastcall *)(__int64))sub_AEF370, (__int64)&v92);
          if ( (*(_BYTE *)(v30 - 17) & 0x20) != 0 )
          {
            v21 = (unsigned __int64)"heapallocsite";
            sub_B9A090(v30 - 24, "heapallocsite", 13, 0);
          }
          sub_B44570(v30 - 24);
          v30 = *(_QWORD *)(v30 + 8);
          if ( v81 + 3 == (__int64 *)v30 )
            goto LABEL_64;
          continue;
        }
      }
LABEL_65:
      v79 = (__int64 *)v79[1];
      if ( a1 + 3 == v79 )
        goto LABEL_66;
    }
    sub_AEE3F0((__int64)&v95, v25);
    if ( (_DWORD)v98 )
    {
      v27 = (v98 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v28 = (const char **)(v96 + 16LL * v27);
      v29 = *v28;
      if ( v26 == *v28 )
      {
LABEL_27:
        if ( v28 != (const char **)(v96 + 16LL * (unsigned int)v98) )
        {
          v21 = (unsigned __int64)v28[1];
          if ( !v21 )
            goto LABEL_61;
          goto LABEL_29;
        }
      }
      else
      {
        v75 = 1;
        while ( v29 != (const char *)-4096LL )
        {
          v77 = v75 + 1;
          v27 = (v98 - 1) & (v75 + v27);
          v28 = (const char **)(v96 + 16LL * v27);
          v29 = *v28;
          if ( v26 == *v28 )
            goto LABEL_27;
          v75 = v77;
        }
      }
    }
    v21 = (unsigned __int64)v26;
LABEL_29:
    if ( (unsigned __int8)(*(_BYTE *)v21 - 5) <= 0x1Fu )
    {
LABEL_30:
      v91 |= v26 != (const char *)v21;
      sub_B994C0(v24, v21);
      goto LABEL_31;
    }
LABEL_61:
    v21 = 0;
    goto LABEL_30;
  }
LABEL_66:
  v52 = (_QWORD *)a1[10];
  if ( v78 != v52 )
  {
    while ( 1 )
    {
      v104 = (__int64 *)v106;
      v53 = 0;
      v105 = 0x800000000LL;
      v54 = sub_B91A00(v52);
      if ( v54 )
        break;
LABEL_96:
      if ( v91 )
      {
        sub_B91A30(v52);
        v70 = v104;
        v73 = &v104[(unsigned int)v105];
        if ( v73 == v104 )
          goto LABEL_98;
        v74 = (const char **)v104;
        do
        {
          v21 = (unsigned __int64)*v74;
          if ( *v74 )
            sub_B979A0(v52, v21);
          ++v74;
        }
        while ( v73 != (__int64 *)v74 );
      }
      v70 = v104;
LABEL_98:
      if ( v70 != (__int64 *)v106 )
        _libc_free(v70, v21);
      v52 = (_QWORD *)v52[1];
      if ( v78 == v52 )
        goto LABEL_101;
    }
    while ( 1 )
    {
      v21 = v53;
      v58 = sub_B91A10(v52, v53);
      v59 = (_BYTE *)v58;
      if ( v58 )
        break;
      v56 = (unsigned int)v105;
      v55 = 0;
      v57 = (unsigned int)v105 + 1LL;
      if ( v57 > HIDWORD(v105) )
      {
LABEL_95:
        v21 = (unsigned __int64)v106;
        v90 = v55;
        sub_C8D5F0(&v104, v106, v57, 8);
        v56 = (unsigned int)v105;
        v55 = v90;
      }
LABEL_71:
      ++v53;
      v104[v56] = (__int64)v55;
      LODWORD(v105) = v105 + 1;
      if ( v54 == v53 )
        goto LABEL_96;
    }
    v21 = v58;
    sub_AEE3F0((__int64)v93, v58);
    v60 = *((unsigned int *)v93 + 6);
    if ( (_DWORD)v60 )
    {
      v21 = v93[1];
      v61 = (v60 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v62 = (const char *)(v21 + 16LL * v61);
      v63 = *(_QWORD *)v62;
      if ( v59 == *(_BYTE **)v62 )
      {
LABEL_75:
        if ( v62 != (const char *)(v21 + 16 * v60) )
        {
          v64 = (_BYTE *)*((_QWORD *)v62 + 1);
          if ( !v64 )
            goto LABEL_78;
          goto LABEL_77;
        }
      }
      else
      {
        v69 = 1;
        while ( v63 != -4096 )
        {
          v76 = v69 + 1;
          v61 = (v60 - 1) & (v69 + v61);
          v62 = (const char *)(v21 + 16LL * v61);
          v63 = *(_QWORD *)v62;
          if ( v59 == *(_BYTE **)v62 )
            goto LABEL_75;
          v69 = v76;
        }
      }
    }
    v64 = v59;
LABEL_77:
    if ( (unsigned __int8)(*v64 - 5) <= 0x1Fu )
    {
      v55 = v64;
      goto LABEL_70;
    }
LABEL_78:
    v55 = 0;
    v64 = 0;
LABEL_70:
    *v94 |= v59 != v64;
    v56 = (unsigned int)v105;
    v57 = (unsigned int)v105 + 1LL;
    if ( v57 > HIDWORD(v105) )
      goto LABEL_95;
    goto LABEL_71;
  }
LABEL_101:
  v71 = v91;
  sub_C7D6A0(v101, 24LL * v103, 8);
  sub_C7D6A0(v96, 16LL * (unsigned int)v98, 8);
  return v71;
}
