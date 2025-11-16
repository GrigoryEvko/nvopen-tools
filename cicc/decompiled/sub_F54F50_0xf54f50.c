// Function: sub_F54F50
// Address: 0xf54f50
//
__int64 __fastcall sub_F54F50(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 (__fastcall *a5)(__int64, __int64),
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64),
        __int64 a8)
{
  unsigned int v10; // r13d
  unsigned __int8 *v11; // r12
  __int64 v13; // rsi
  __int64 v14; // rax
  bool v15; // cc
  __int64 *v16; // r8
  __int64 *v17; // r15
  __int64 v18; // r12
  unsigned __int8 **v19; // rax
  unsigned __int8 **v20; // rdx
  __int64 *v21; // r15
  __int64 v22; // r12
  unsigned __int8 **v23; // rax
  unsigned __int8 **v24; // rdx
  _BYTE *v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // ebx
  __int64 *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // ebx
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 *v39; // r13
  unsigned __int8 *v40; // r15
  unsigned __int8 **v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int8 **v45; // rax
  __int64 *v46; // rcx
  bool v47; // zf
  __int64 *v48; // r13
  __int64 v49; // r15
  bool v50; // r12
  unsigned __int8 *v51; // r14
  __int64 v52; // rdx
  bool v53; // al
  unsigned __int8 **v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  unsigned __int8 **v58; // rax
  __int64 v59; // rsi
  unsigned int v60; // esi
  __int64 v61; // rax
  unsigned __int8 *v62; // [rsp+0h] [rbp-160h]
  __int64 v64; // [rsp+30h] [rbp-130h]
  unsigned __int8 v65; // [rsp+30h] [rbp-130h]
  __int64 v66; // [rsp+38h] [rbp-128h]
  unsigned __int8 *v67; // [rsp+38h] [rbp-128h]
  unsigned __int8 v68; // [rsp+38h] [rbp-128h]
  __int64 v69; // [rsp+38h] [rbp-128h]
  bool v70; // [rsp+38h] [rbp-128h]
  __int64 v73; // [rsp+50h] [rbp-110h]
  __int64 *v74; // [rsp+58h] [rbp-108h]
  __int64 *v75; // [rsp+58h] [rbp-108h]
  __int64 *v76; // [rsp+58h] [rbp-108h]
  __int64 *v77; // [rsp+58h] [rbp-108h]
  unsigned __int8 *v78; // [rsp+68h] [rbp-F8h] BYREF
  _BYTE *v79; // [rsp+70h] [rbp-F0h]
  __int64 v80; // [rsp+78h] [rbp-E8h]
  __int64 v81; // [rsp+80h] [rbp-E0h]
  __int64 v82; // [rsp+88h] [rbp-D8h]
  __int64 *v83; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+98h] [rbp-C8h]
  _BYTE v85[16]; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v86; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-A8h]
  _BYTE v88[16]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v89; // [rsp+D0h] [rbp-90h] BYREF
  unsigned __int8 **v90; // [rsp+D8h] [rbp-88h]
  __int64 v91; // [rsp+E0h] [rbp-80h]
  int v92; // [rsp+E8h] [rbp-78h]
  char v93; // [rsp+ECh] [rbp-74h]
  char v94; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v95; // [rsp+100h] [rbp-60h] BYREF
  unsigned __int8 **v96; // [rsp+108h] [rbp-58h]
  __int64 v97; // [rsp+110h] [rbp-50h]
  int v98; // [rsp+118h] [rbp-48h]
  char v99; // [rsp+11Ch] [rbp-44h]
  char v100; // [rsp+120h] [rbp-40h] BYREF

  v10 = 0;
  v11 = (unsigned __int8 *)a1;
  v13 = a1;
  v86 = (__int64 *)v88;
  v83 = (__int64 *)v85;
  v84 = 0x100000000LL;
  v87 = 0x100000000LL;
  sub_AE7A50((__int64)&v83, a1, (__int64)&v86);
  v14 = (unsigned int)v84;
  if ( !((unsigned int)v87 | (unsigned int)v84) )
    goto LABEL_28;
  v15 = *a2 <= 0x1Cu;
  v93 = 1;
  v90 = (unsigned __int8 **)&v94;
  v89 = 0;
  v91 = 1;
  v92 = 0;
  v95 = 0;
  v96 = (unsigned __int8 **)&v100;
  v97 = 1;
  v98 = 0;
  v99 = 1;
  if ( v15 )
  {
LABEL_3:
    v16 = v83;
    v74 = &v83[v14];
    if ( v74 == v83 )
      goto LABEL_12;
    goto LABEL_4;
  }
  v64 = sub_B46B10(a1, 0);
  v13 = v64;
  v76 = &v83[(unsigned int)v84];
  if ( v83 == v76 )
  {
    v46 = v86;
    v77 = &v86[(unsigned int)v87];
    if ( v86 == v77 )
      goto LABEL_21;
LABEL_65:
    v47 = v64 == a3;
    v65 = v10;
    v48 = v46;
    v49 = a4;
    v62 = v11;
    v50 = v47;
    while ( 1 )
    {
      v51 = (unsigned __int8 *)*v48;
      v52 = **(_QWORD **)(*v48 + 16);
      if ( *(_BYTE *)v52 == 85 )
      {
        v59 = *(_QWORD *)(v52 - 32);
        if ( v59 )
        {
          if ( !*(_BYTE *)v59 && *(_QWORD *)(v59 + 24) == *(_QWORD *)(v52 + 80) && (*(_BYTE *)(v59 + 33) & 0x20) != 0 )
          {
            v60 = *(_DWORD *)(v59 + 36);
            if ( v60 > 0x45 )
            {
              if ( v60 == 71 )
              {
LABEL_87:
                v69 = **(_QWORD **)(*v48 + 16);
                v61 = sub_B46B10(v69, 0);
                v52 = v69;
                v53 = v50 && a3 == v61;
                if ( v53 )
                {
LABEL_88:
                  v70 = v53;
                  sub_B14260(v51);
                  v13 = (__int64)v51;
                  sub_AA8740(*(_QWORD *)(a3 + 40), (__int64)v51, a3);
                  v65 = v70;
                  goto LABEL_66;
                }
                goto LABEL_69;
              }
            }
            else if ( v60 > 0x43 )
            {
              goto LABEL_87;
            }
          }
        }
      }
      v53 = v50 && a3 == **(_QWORD **)(*v48 + 16);
      if ( v53 )
        goto LABEL_88;
LABEL_69:
      v13 = a3;
      if ( (unsigned __int8)sub_B19DB0(v49, a3, v52) )
      {
LABEL_66:
        if ( v77 == ++v48 )
          goto LABEL_76;
      }
      else
      {
        if ( !v99 )
          goto LABEL_80;
        v58 = v96;
        v13 = HIDWORD(v97);
        v54 = &v96[HIDWORD(v97)];
        if ( v96 != v54 )
        {
          while ( v51 != *v58 )
          {
            if ( v54 == ++v58 )
              goto LABEL_74;
          }
          goto LABEL_66;
        }
LABEL_74:
        if ( HIDWORD(v97) >= (unsigned int)v97 )
        {
LABEL_80:
          v13 = (__int64)v51;
          sub_C8CC70((__int64)&v95, (__int64)v51, (__int64)v54, v55, v56, v57);
          goto LABEL_66;
        }
        v13 = (unsigned int)(HIDWORD(v97) + 1);
        ++v48;
        ++HIDWORD(v97);
        *v54 = v51;
        ++v95;
        if ( v77 == v48 )
        {
LABEL_76:
          v10 = v65;
          v11 = v62;
          v14 = (unsigned int)v84;
          goto LABEL_3;
        }
      }
    }
  }
  v68 = 0;
  v39 = v83;
  do
  {
    while ( 1 )
    {
      v40 = (unsigned __int8 *)*v39;
      if ( v64 == a3 && v64 == sub_B46B10(*v39, 0) )
      {
        v13 = v64;
        sub_B44530(v40, v64);
        v68 = 1;
        goto LABEL_55;
      }
      v13 = a3;
      if ( !(unsigned __int8)sub_B19DB0(a4, a3, (__int64)v40) )
        break;
LABEL_55:
      if ( v76 == ++v39 )
        goto LABEL_64;
    }
    if ( !v93 )
      goto LABEL_77;
    v45 = v90;
    v13 = HIDWORD(v91);
    v41 = &v90[HIDWORD(v91)];
    if ( v90 != v41 )
    {
      while ( v40 != *v45 )
      {
        if ( v41 == ++v45 )
          goto LABEL_62;
      }
      goto LABEL_55;
    }
LABEL_62:
    if ( HIDWORD(v91) >= (unsigned int)v91 )
    {
LABEL_77:
      v13 = (__int64)v40;
      sub_C8CC70((__int64)&v89, (__int64)v40, (__int64)v41, v42, v43, v44);
      goto LABEL_55;
    }
    v13 = (unsigned int)(HIDWORD(v91) + 1);
    ++v39;
    ++HIDWORD(v91);
    *v41 = v40;
    ++v89;
  }
  while ( v76 != v39 );
LABEL_64:
  v46 = v86;
  v10 = v68;
  v11 = (unsigned __int8 *)a1;
  v77 = &v86[(unsigned int)v87];
  if ( v77 != v86 )
    goto LABEL_65;
  v16 = v83;
  v74 = &v83[(unsigned int)v84];
  if ( v74 == v83 )
    goto LABEL_21;
LABEL_4:
  v66 = (__int64)v11;
  v17 = v16;
  while ( 2 )
  {
    while ( 2 )
    {
      v18 = *v17;
      if ( v93 )
      {
        v19 = v90;
        v20 = &v90[HIDWORD(v91)];
        if ( v90 == v20 )
          goto LABEL_34;
        while ( (unsigned __int8 *)v18 != *v19 )
        {
          if ( v20 == ++v19 )
            goto LABEL_34;
        }
        goto LABEL_10;
      }
      v13 = *v17;
      if ( sub_C8CA60((__int64)&v89, v18) )
        goto LABEL_10;
LABEL_34:
      v13 = v18;
      v26 = (_BYTE *)a5(a6, v18);
      v80 = v27;
      v28 = (unsigned __int8)v27;
      v79 = v26;
      if ( !(_BYTE)v27 )
        goto LABEL_10;
      sub_B59720(v18, v66, a2);
      v29 = (__int64 *)(*((_QWORD *)v79 + 1) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*((_QWORD *)v79 + 1) & 4) != 0 )
        v29 = (__int64 *)*v29;
      v13 = (__int64)v79;
      v30 = sub_B9F6F0(v29, v79);
      v31 = v18 + 32 * (2LL - (*(_DWORD *)(v18 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v31 )
      {
        v32 = *(_QWORD *)(v31 + 8);
        **(_QWORD **)(v31 + 16) = v32;
        if ( v32 )
          *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
      }
      *(_QWORD *)v31 = v30;
      v10 = v28;
      if ( !v30 )
      {
LABEL_10:
        if ( v74 == ++v17 )
          goto LABEL_11;
        continue;
      }
      break;
    }
    v33 = *(_QWORD *)(v30 + 16);
    *(_QWORD *)(v31 + 8) = v33;
    if ( v33 )
    {
      v13 = v31 + 8;
      *(_QWORD *)(v33 + 16) = v31 + 8;
    }
    *(_QWORD *)(v31 + 16) = v30 + 16;
    v10 = v28;
    ++v17;
    *(_QWORD *)(v30 + 16) = v31;
    if ( v74 != v17 )
      continue;
    break;
  }
LABEL_11:
  v11 = (unsigned __int8 *)v66;
LABEL_12:
  v75 = &v86[(unsigned int)v87];
  if ( v75 != v86 )
  {
    v67 = v11;
    v21 = v86;
    while ( 1 )
    {
      v22 = *v21;
      if ( v99 )
      {
        v23 = v96;
        v24 = &v96[HIDWORD(v97)];
        if ( v96 != v24 )
        {
          while ( (unsigned __int8 *)v22 != *v23 )
          {
            if ( v24 == ++v23 )
              goto LABEL_46;
          }
          goto LABEL_19;
        }
LABEL_46:
        v13 = v22;
        v34 = a7(a8, v22);
        v82 = v35;
        v36 = (unsigned __int8)v35;
        v81 = v34;
        if ( !(_BYTE)v35 )
          goto LABEL_19;
        v73 = v81;
        sub_B13360(v22, v67, a2, 0);
        sub_B11F20(&v78, v73);
        v37 = *(_QWORD *)(v22 + 80);
        v38 = v22 + 80;
        if ( v37 )
        {
          sub_B91220(v22 + 80, v37);
          v38 = v22 + 80;
        }
        v13 = (__int64)v78;
        *(_QWORD *)(v22 + 80) = v78;
        if ( v13 )
          sub_B976B0((__int64)&v78, (unsigned __int8 *)v13, v38);
        v10 = v36;
        if ( v75 == ++v21 )
        {
LABEL_20:
          v11 = v67;
          break;
        }
      }
      else
      {
        v13 = *v21;
        if ( !sub_C8CA60((__int64)&v95, v22) )
          goto LABEL_46;
LABEL_19:
        if ( v75 == ++v21 )
          goto LABEL_20;
      }
    }
  }
LABEL_21:
  if ( HIDWORD(v91) != v92 || HIDWORD(v97) != v98 )
  {
    v10 = 1;
    sub_F54ED0(v11);
  }
  if ( !v99 )
    _libc_free(v96, v13);
  if ( !v93 )
    _libc_free(v90, v13);
LABEL_28:
  if ( v86 != (__int64 *)v88 )
    _libc_free(v86, v13);
  if ( v83 != (__int64 *)v85 )
    _libc_free(v83, v13);
  return v10;
}
