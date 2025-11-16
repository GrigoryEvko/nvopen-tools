// Function: sub_1591060
// Address: 0x1591060
//
__int64 __fastcall sub_1591060(__int64 a1, int a2, __int64 a3, int a4)
{
  unsigned int v6; // r14d
  unsigned int v8; // edx
  __int64 v9; // r8
  bool v10; // cc
  char v11; // al
  __int64 v12; // rdi
  unsigned int v13; // edx
  __int64 v14; // r8
  char v15; // al
  bool v16; // al
  unsigned int v17; // r12d
  __int64 v18; // rax
  bool v19; // al
  unsigned int v20; // r12d
  __int64 v21; // rcx
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // rdx
  unsigned int v25; // eax
  unsigned int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // r12
  unsigned int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // rdx
  unsigned int v32; // r12d
  __int64 v33; // rcx
  unsigned int v34; // eax
  unsigned int v35; // eax
  __int64 v36; // rdi
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 v39; // r12
  unsigned int v40; // eax
  unsigned int v41; // eax
  __int64 v42; // [rsp+18h] [rbp-158h]
  __int64 v43; // [rsp+18h] [rbp-158h]
  __int64 v44; // [rsp+18h] [rbp-158h]
  __int64 v45; // [rsp+18h] [rbp-158h]
  unsigned int v46; // [rsp+20h] [rbp-150h]
  char v47; // [rsp+20h] [rbp-150h]
  int v48; // [rsp+20h] [rbp-150h]
  unsigned int v49; // [rsp+20h] [rbp-150h]
  char v50; // [rsp+20h] [rbp-150h]
  int v51; // [rsp+20h] [rbp-150h]
  unsigned int v52; // [rsp+20h] [rbp-150h]
  unsigned int v53; // [rsp+20h] [rbp-150h]
  __int64 v55; // [rsp+30h] [rbp-140h] BYREF
  unsigned int v56; // [rsp+38h] [rbp-138h]
  __int64 v57; // [rsp+40h] [rbp-130h] BYREF
  unsigned int v58; // [rsp+48h] [rbp-128h]
  unsigned __int64 v59; // [rsp+50h] [rbp-120h] BYREF
  unsigned int v60; // [rsp+58h] [rbp-118h]
  unsigned __int64 v61; // [rsp+60h] [rbp-110h] BYREF
  unsigned int v62; // [rsp+68h] [rbp-108h]
  unsigned __int64 v63; // [rsp+70h] [rbp-100h] BYREF
  unsigned int v64; // [rsp+78h] [rbp-F8h]
  __int64 v65; // [rsp+80h] [rbp-F0h] BYREF
  unsigned int v66; // [rsp+88h] [rbp-E8h]
  __int64 v67; // [rsp+90h] [rbp-E0h]
  unsigned int v68; // [rsp+98h] [rbp-D8h]
  __int64 v69; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v70; // [rsp+A8h] [rbp-C8h]
  __int64 v71; // [rsp+B0h] [rbp-C0h]
  unsigned int v72; // [rsp+B8h] [rbp-B8h]
  __int64 v73; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned int v74; // [rsp+C8h] [rbp-A8h]
  __int64 v75; // [rsp+D0h] [rbp-A0h]
  unsigned int v76; // [rsp+D8h] [rbp-98h]
  __int64 v77; // [rsp+E0h] [rbp-90h] BYREF
  unsigned int v78; // [rsp+E8h] [rbp-88h]
  __int64 v79; // [rsp+F0h] [rbp-80h]
  unsigned int v80; // [rsp+F8h] [rbp-78h]
  __int64 v81; // [rsp+100h] [rbp-70h] BYREF
  unsigned int v82; // [rsp+108h] [rbp-68h]
  __int64 v83; // [rsp+110h] [rbp-60h]
  unsigned int v84; // [rsp+118h] [rbp-58h]
  __int64 v85; // [rsp+120h] [rbp-50h] BYREF
  unsigned int v86; // [rsp+128h] [rbp-48h]
  __int64 v87; // [rsp+130h] [rbp-40h]
  unsigned int v88; // [rsp+138h] [rbp-38h]

  v6 = *(_DWORD *)(a3 + 8);
  sub_15897D0((__int64)&v65, v6, 1);
  if ( a2 == 13 )
  {
    v82 = *(_DWORD *)(a3 + 8);
    if ( v82 > 0x40 )
      sub_16A4FD0(&v81, a3);
    else
      v81 = *(_QWORD *)a3;
    sub_16A7490(&v81, 1);
    v13 = v82;
    v14 = v81;
    v82 = 0;
    v10 = *(_DWORD *)(a3 + 24) <= 0x40u;
    v86 = v13;
    v85 = v81;
    if ( v10 )
    {
      v15 = *(_QWORD *)(a3 + 16) == v81;
    }
    else
    {
      v43 = v81;
      v49 = v13;
      v15 = sub_16A5220(a3 + 16, &v85);
      v14 = v43;
      v13 = v49;
    }
    if ( v13 > 0x40 )
    {
      if ( v14 )
      {
        v50 = v15;
        j_j___libc_free_0_0(v14);
        v15 = v50;
        if ( v82 > 0x40 )
        {
          if ( v81 )
          {
            j_j___libc_free_0_0(v81);
            v15 = v50;
          }
        }
      }
    }
    if ( v15 )
    {
      if ( *(_DWORD *)(a3 + 8) <= 0x40u )
      {
        v16 = *(_QWORD *)a3 == 0;
      }
      else
      {
        v51 = *(_DWORD *)(a3 + 8);
        v16 = v51 == (unsigned int)sub_16A57B0(a3);
      }
      if ( v16 )
        goto LABEL_170;
    }
    if ( (a4 & 1) != 0 )
    {
      v64 = v6;
      if ( v6 <= 0x40 )
        v63 = 0;
      else
        sub_16A4EF0(&v63, 0, 0);
      sub_158A9F0((__int64)&v61, a3);
      sub_15898E0((__int64)&v69, (__int64)&v61, (__int64 *)&v63);
      sub_1590E70((__int64)&v81, (__int64)&v65);
      sub_1590E70((__int64)&v85, (__int64)&v69);
      sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
      sub_1590E70((__int64)&v73, (__int64)&v77);
      if ( v80 > 0x40 && v79 )
        j_j___libc_free_0_0(v79);
      if ( v78 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      if ( v86 > 0x40 && v85 )
        j_j___libc_free_0_0(v85);
      if ( v84 > 0x40 && v83 )
        j_j___libc_free_0_0(v83);
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
      if ( v66 > 0x40 && v65 )
        j_j___libc_free_0_0(v65);
      v65 = v73;
      v25 = v74;
      v74 = 0;
      v66 = v25;
      if ( v68 > 0x40 && v67 )
      {
        j_j___libc_free_0_0(v67);
        v67 = v75;
        v68 = v76;
        if ( v74 > 0x40 && v73 )
          j_j___libc_free_0_0(v73);
      }
      else
      {
        v67 = v75;
        v68 = v76;
      }
      if ( v72 > 0x40 && v71 )
        j_j___libc_free_0_0(v71);
      if ( v70 > 0x40 && v69 )
        j_j___libc_free_0_0(v69);
      if ( v62 > 0x40 && v61 )
        j_j___libc_free_0_0(v61);
      if ( v64 > 0x40 && v63 )
        j_j___libc_free_0_0(v63);
    }
    if ( (a4 & 2) == 0 )
      goto LABEL_26;
    sub_158ACE0((__int64)&v55, a3);
    sub_158ABC0((__int64)&v57, a3);
    v17 = v58;
    v18 = 1LL << ((unsigned __int8)v58 - 1);
    if ( v58 <= 0x40 )
    {
      if ( (v57 & v18) != 0 )
        goto LABEL_128;
      v19 = v57 == 0;
    }
    else
    {
      if ( (*(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6)) & v18) != 0 )
        goto LABEL_128;
      v19 = v17 == (unsigned int)sub_16A57B0(&v57);
    }
    if ( v19 )
      goto LABEL_128;
    v20 = v6 - 1;
    v64 = v6;
    v21 = 1LL << ((unsigned __int8)v6 - 1);
    if ( v6 > 0x40 )
    {
      v45 = 1LL << ((unsigned __int8)v6 - 1);
      sub_16A4EF0(&v63, 0, 0);
      if ( v64 <= 0x40 )
        v63 |= v45;
      else
        *(_QWORD *)(v63 + 8LL * (v20 >> 6)) |= v45;
      v60 = v6;
      sub_16A4EF0(&v59, 0, 0);
      v21 = 1LL << ((unsigned __int8)v6 - 1);
      if ( v60 > 0x40 )
      {
        *(_QWORD *)(v59 + 8LL * (v20 >> 6)) |= 1LL << ((unsigned __int8)v6 - 1);
LABEL_87:
        sub_16A7200(&v59, &v57);
        v22 = v60;
        v60 = 0;
        v62 = v22;
        v61 = v59;
        sub_15898E0((__int64)&v69, (__int64)&v61, (__int64 *)&v63);
        sub_1590E70((__int64)&v81, (__int64)&v65);
        sub_1590E70((__int64)&v85, (__int64)&v69);
        sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
        sub_1590E70((__int64)&v73, (__int64)&v77);
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        if ( v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        if ( v88 > 0x40 && v87 )
          j_j___libc_free_0_0(v87);
        if ( v86 > 0x40 && v85 )
          j_j___libc_free_0_0(v85);
        if ( v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
        if ( v82 > 0x40 && v81 )
          j_j___libc_free_0_0(v81);
        if ( v66 > 0x40 && v65 )
          j_j___libc_free_0_0(v65);
        v65 = v73;
        v23 = v74;
        v74 = 0;
        v66 = v23;
        if ( v68 > 0x40 && v67 )
        {
          j_j___libc_free_0_0(v67);
          v67 = v75;
          v68 = v76;
          if ( v74 > 0x40 && v73 )
            j_j___libc_free_0_0(v73);
        }
        else
        {
          v67 = v75;
          v68 = v76;
        }
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( v62 > 0x40 && v61 )
          j_j___libc_free_0_0(v61);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
LABEL_128:
        v24 = v55;
        if ( v56 > 0x40 )
          v24 = *(_QWORD *)(v55 + 8LL * ((v56 - 1) >> 6));
        if ( (v24 & (1LL << ((unsigned __int8)v56 - 1))) == 0 )
          goto LABEL_131;
        v62 = v6;
        v53 = v6 - 1;
        v39 = 1LL << ((unsigned __int8)v6 - 1);
        if ( v6 > 0x40 )
        {
          sub_16A4EF0(&v61, 0, 0);
          if ( v62 <= 0x40 )
            v61 |= v39;
          else
            *(_QWORD *)(v61 + 8LL * (v53 >> 6)) |= v39;
          sub_16A7200(&v61, &v55);
          v60 = v6;
          v64 = v62;
          v63 = v61;
          v62 = 0;
          sub_16A4EF0(&v59, 0, 0);
          if ( v60 > 0x40 )
          {
            *(_QWORD *)(v59 + 8LL * (v53 >> 6)) |= v39;
            goto LABEL_369;
          }
        }
        else
        {
          v61 = 1LL << ((unsigned __int8)v6 - 1);
          sub_16A7200(&v61, &v55);
          v40 = v62;
          v60 = v6;
          v62 = 0;
          v64 = v40;
          v59 = 0;
          v63 = v61;
        }
        v59 |= v39;
LABEL_369:
        sub_15898E0((__int64)&v69, (__int64)&v59, (__int64 *)&v63);
        sub_1590E70((__int64)&v81, (__int64)&v65);
        sub_1590E70((__int64)&v85, (__int64)&v69);
        sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
        sub_1590E70((__int64)&v73, (__int64)&v77);
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        if ( v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        if ( v88 > 0x40 && v87 )
          j_j___libc_free_0_0(v87);
        if ( v86 > 0x40 && v85 )
          j_j___libc_free_0_0(v85);
        if ( v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
        if ( v82 > 0x40 && v81 )
          j_j___libc_free_0_0(v81);
        if ( v66 > 0x40 && v65 )
          j_j___libc_free_0_0(v65);
        v65 = v73;
        v41 = v74;
        v74 = 0;
        v66 = v41;
        if ( v68 > 0x40 && v67 )
        {
          j_j___libc_free_0_0(v67);
          v67 = v75;
          v68 = v76;
          if ( v74 > 0x40 && v73 )
            j_j___libc_free_0_0(v73);
        }
        else
        {
          v67 = v75;
          v68 = v76;
        }
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
        if ( v62 <= 0x40 )
          goto LABEL_131;
        v36 = v61;
        if ( !v61 )
          goto LABEL_131;
LABEL_309:
        j_j___libc_free_0_0(v36);
LABEL_131:
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        if ( v56 > 0x40 && v55 )
          j_j___libc_free_0_0(v55);
LABEL_26:
        *(_DWORD *)(a1 + 8) = v66;
        *(_QWORD *)a1 = v65;
        *(_DWORD *)(a1 + 24) = v68;
        *(_QWORD *)(a1 + 16) = v67;
        return a1;
      }
    }
    else
    {
      v63 = 1LL << ((unsigned __int8)v6 - 1);
      v60 = v6;
      v59 = 0;
    }
    v59 |= v21;
    goto LABEL_87;
  }
  if ( a2 != 15 )
  {
    if ( a2 != 11 )
    {
      sub_15897D0(a1, v6, 0);
      goto LABEL_5;
    }
    v82 = *(_DWORD *)(a3 + 8);
    if ( v82 > 0x40 )
      sub_16A4FD0(&v81, a3);
    else
      v81 = *(_QWORD *)a3;
    sub_16A7490(&v81, 1);
    v8 = v82;
    v9 = v81;
    v82 = 0;
    v10 = *(_DWORD *)(a3 + 24) <= 0x40u;
    v86 = v8;
    v85 = v81;
    if ( v10 )
    {
      v11 = *(_QWORD *)(a3 + 16) == v81;
    }
    else
    {
      v42 = v81;
      v46 = v8;
      v11 = sub_16A5220(a3 + 16, &v85);
      v9 = v42;
      v8 = v46;
    }
    if ( v8 > 0x40 )
    {
      if ( v9 )
      {
        v47 = v11;
        j_j___libc_free_0_0(v9);
        v11 = v47;
        if ( v82 > 0x40 )
        {
          if ( v81 )
          {
            j_j___libc_free_0_0(v81);
            v11 = v47;
          }
        }
      }
    }
    if ( !v11 )
      goto LABEL_24;
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      v48 = *(_DWORD *)(a3 + 8);
      if ( v48 != (unsigned int)sub_16A57B0(a3) )
        goto LABEL_24;
      goto LABEL_170;
    }
    if ( *(_QWORD *)a3 )
    {
LABEL_24:
      if ( (a4 & 1) != 0 )
      {
        sub_158A9F0((__int64)&v61, a3);
        if ( v62 <= 0x40 )
          v61 = ~v61 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v62);
        else
          sub_16A8F40(&v61);
        sub_16A7400(&v61);
        v37 = v62;
        v60 = v6;
        v62 = 0;
        v64 = v37;
        v63 = v61;
        if ( v6 > 0x40 )
          sub_16A4EF0(&v59, 0, 0);
        else
          v59 = 0;
        sub_15898E0((__int64)&v69, (__int64)&v59, (__int64 *)&v63);
        sub_1590E70((__int64)&v81, (__int64)&v65);
        sub_1590E70((__int64)&v85, (__int64)&v69);
        sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
        sub_1590E70((__int64)&v73, (__int64)&v77);
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        if ( v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        if ( v88 > 0x40 && v87 )
          j_j___libc_free_0_0(v87);
        if ( v86 > 0x40 && v85 )
          j_j___libc_free_0_0(v85);
        if ( v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
        if ( v82 > 0x40 && v81 )
          j_j___libc_free_0_0(v81);
        if ( v66 > 0x40 && v65 )
          j_j___libc_free_0_0(v65);
        v65 = v73;
        v38 = v74;
        v74 = 0;
        v66 = v38;
        if ( v68 > 0x40 && v67 )
        {
          j_j___libc_free_0_0(v67);
          v67 = v75;
          v68 = v76;
          if ( v74 > 0x40 && v73 )
            j_j___libc_free_0_0(v73);
        }
        else
        {
          v67 = v75;
          v68 = v76;
        }
        if ( v72 > 0x40 && v71 )
          j_j___libc_free_0_0(v71);
        if ( v70 > 0x40 && v69 )
          j_j___libc_free_0_0(v69);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        if ( v64 > 0x40 && v63 )
          j_j___libc_free_0_0(v63);
        if ( v62 > 0x40 && v61 )
          j_j___libc_free_0_0(v61);
      }
      if ( (a4 & 2) == 0 )
        goto LABEL_26;
      sub_158ACE0((__int64)&v55, a3);
      sub_158ABC0((__int64)&v57, a3);
      v26 = v58;
      v27 = 1LL << ((unsigned __int8)v58 - 1);
      if ( v58 <= 0x40 )
      {
        if ( (v57 & v27) != 0 || !v57 )
          goto LABEL_263;
      }
      else if ( (*(_QWORD *)(v57 + 8LL * ((v58 - 1) >> 6)) & v27) != 0 || v26 == (unsigned int)sub_16A57B0(&v57) )
      {
        goto LABEL_263;
      }
      v62 = v6;
      v52 = v6 - 1;
      v28 = 1LL << ((unsigned __int8)v6 - 1);
      if ( v6 > 0x40 )
      {
        sub_16A4EF0(&v61, 0, 0);
        if ( v62 <= 0x40 )
          v61 |= v28;
        else
          *(_QWORD *)(v61 + 8LL * (v52 >> 6)) |= v28;
        sub_16A7590(&v61, &v57);
        v60 = v6;
        v64 = v62;
        v63 = v61;
        v62 = 0;
        sub_16A4EF0(&v59, 0, 0);
        if ( v60 > 0x40 )
        {
          *(_QWORD *)(v59 + 8LL * (v52 >> 6)) |= v28;
LABEL_222:
          sub_15898E0((__int64)&v69, (__int64)&v59, (__int64 *)&v63);
          sub_1590E70((__int64)&v81, (__int64)&v65);
          sub_1590E70((__int64)&v85, (__int64)&v69);
          sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
          sub_1590E70((__int64)&v73, (__int64)&v77);
          if ( v80 > 0x40 && v79 )
            j_j___libc_free_0_0(v79);
          if ( v78 > 0x40 && v77 )
            j_j___libc_free_0_0(v77);
          if ( v88 > 0x40 && v87 )
            j_j___libc_free_0_0(v87);
          if ( v86 > 0x40 && v85 )
            j_j___libc_free_0_0(v85);
          if ( v84 > 0x40 && v83 )
            j_j___libc_free_0_0(v83);
          if ( v82 > 0x40 && v81 )
            j_j___libc_free_0_0(v81);
          if ( v66 > 0x40 && v65 )
            j_j___libc_free_0_0(v65);
          v65 = v73;
          v30 = v74;
          v74 = 0;
          v66 = v30;
          if ( v68 > 0x40 && v67 )
          {
            j_j___libc_free_0_0(v67);
            v67 = v75;
            v68 = v76;
            if ( v74 > 0x40 && v73 )
              j_j___libc_free_0_0(v73);
          }
          else
          {
            v67 = v75;
            v68 = v76;
          }
          if ( v72 > 0x40 && v71 )
            j_j___libc_free_0_0(v71);
          if ( v70 > 0x40 && v69 )
            j_j___libc_free_0_0(v69);
          if ( v60 > 0x40 && v59 )
            j_j___libc_free_0_0(v59);
          if ( v64 > 0x40 && v63 )
            j_j___libc_free_0_0(v63);
          if ( v62 > 0x40 && v61 )
            j_j___libc_free_0_0(v61);
LABEL_263:
          v31 = v55;
          if ( v56 > 0x40 )
            v31 = *(_QWORD *)(v55 + 8LL * ((v56 - 1) >> 6));
          if ( (v31 & (1LL << ((unsigned __int8)v56 - 1))) == 0 )
            goto LABEL_131;
          v32 = v6 - 1;
          v64 = v6;
          v33 = 1LL << ((unsigned __int8)v6 - 1);
          if ( v6 > 0x40 )
          {
            v44 = 1LL << ((unsigned __int8)v6 - 1);
            sub_16A4EF0(&v63, 0, 0);
            if ( v64 <= 0x40 )
              v63 |= v44;
            else
              *(_QWORD *)(v63 + 8LL * (v32 >> 6)) |= v44;
            v60 = v6;
            sub_16A4EF0(&v59, 0, 0);
            v33 = 1LL << ((unsigned __int8)v6 - 1);
            if ( v60 > 0x40 )
            {
              *(_QWORD *)(v59 + 8LL * (v32 >> 6)) |= 1LL << ((unsigned __int8)v6 - 1);
              goto LABEL_269;
            }
          }
          else
          {
            v63 = 1LL << ((unsigned __int8)v6 - 1);
            v60 = v6;
            v59 = 0;
          }
          v59 |= v33;
LABEL_269:
          sub_16A7590(&v59, &v55);
          v34 = v60;
          v60 = 0;
          v62 = v34;
          v61 = v59;
          sub_15898E0((__int64)&v69, (__int64)&v61, (__int64 *)&v63);
          sub_1590E70((__int64)&v81, (__int64)&v65);
          sub_1590E70((__int64)&v85, (__int64)&v69);
          sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
          sub_1590E70((__int64)&v73, (__int64)&v77);
          if ( v80 > 0x40 && v79 )
            j_j___libc_free_0_0(v79);
          if ( v78 > 0x40 && v77 )
            j_j___libc_free_0_0(v77);
          if ( v88 > 0x40 && v87 )
            j_j___libc_free_0_0(v87);
          if ( v86 > 0x40 && v85 )
            j_j___libc_free_0_0(v85);
          if ( v84 > 0x40 && v83 )
            j_j___libc_free_0_0(v83);
          if ( v82 > 0x40 && v81 )
            j_j___libc_free_0_0(v81);
          if ( v66 > 0x40 && v65 )
            j_j___libc_free_0_0(v65);
          v65 = v73;
          v35 = v74;
          v74 = 0;
          v66 = v35;
          if ( v68 > 0x40 && v67 )
          {
            j_j___libc_free_0_0(v67);
            v67 = v75;
            v68 = v76;
            if ( v74 > 0x40 && v73 )
              j_j___libc_free_0_0(v73);
          }
          else
          {
            v67 = v75;
            v68 = v76;
          }
          if ( v72 > 0x40 && v71 )
            j_j___libc_free_0_0(v71);
          if ( v70 > 0x40 && v69 )
            j_j___libc_free_0_0(v69);
          if ( v62 > 0x40 && v61 )
            j_j___libc_free_0_0(v61);
          if ( v60 > 0x40 && v59 )
            j_j___libc_free_0_0(v59);
          if ( v64 <= 0x40 )
            goto LABEL_131;
          v36 = v63;
          if ( !v63 )
            goto LABEL_131;
          goto LABEL_309;
        }
      }
      else
      {
        v61 = 1LL << ((unsigned __int8)v6 - 1);
        sub_16A7590(&v61, &v57);
        v29 = v62;
        v60 = v6;
        v62 = 0;
        v64 = v29;
        v59 = 0;
        v63 = v61;
      }
      v59 |= v28;
      goto LABEL_222;
    }
LABEL_170:
    sub_15897D0(a1, v6, 1);
    goto LABEL_5;
  }
  if ( a4 == 3 )
  {
    sub_1591060(&v73, 15, a3, 1);
    sub_1591060(&v69, 15, a3, 2);
    sub_1590E70((__int64)&v81, (__int64)&v69);
    sub_1590E70((__int64)&v85, (__int64)&v73);
    sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
    sub_1590E70(a1, (__int64)&v77);
    if ( v80 > 0x40 && v79 )
      j_j___libc_free_0_0(v79);
    if ( v78 > 0x40 && v77 )
      j_j___libc_free_0_0(v77);
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0(v87);
    if ( v86 > 0x40 && v85 )
      j_j___libc_free_0_0(v85);
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    if ( v82 > 0x40 && v81 )
      j_j___libc_free_0_0(v81);
    if ( v72 > 0x40 && v71 )
      j_j___libc_free_0_0(v71);
    if ( v70 > 0x40 && v69 )
      j_j___libc_free_0_0(v69);
    if ( v76 > 0x40 && v75 )
      j_j___libc_free_0_0(v75);
    if ( v74 <= 0x40 )
      goto LABEL_5;
    v12 = v73;
    if ( !v73 )
      goto LABEL_5;
  }
  else
  {
    HIDWORD(v59) = v6;
    LOBYTE(v59) = a4 == 1;
    if ( a4 == 1 )
    {
      sub_158A9F0((__int64)&v85, a3);
      sub_1589910(a1, (__int64)&v59, (__int64)&v85);
      if ( v86 <= 0x40 )
        goto LABEL_5;
      v12 = v85;
      if ( !v85 )
        goto LABEL_5;
    }
    else
    {
      sub_158ABC0((__int64)&v63, a3);
      sub_1589910((__int64)&v73, (__int64)&v59, (__int64)&v63);
      sub_158ACE0((__int64)&v61, a3);
      sub_1589910((__int64)&v69, (__int64)&v59, (__int64)&v61);
      sub_1590E70((__int64)&v81, (__int64)&v69);
      sub_1590E70((__int64)&v85, (__int64)&v73);
      sub_158C3A0((__int64)&v77, (__int64)&v81, (__int64)&v85);
      sub_1590E70(a1, (__int64)&v77);
      if ( v80 > 0x40 && v79 )
        j_j___libc_free_0_0(v79);
      if ( v78 > 0x40 && v77 )
        j_j___libc_free_0_0(v77);
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      if ( v86 > 0x40 && v85 )
        j_j___libc_free_0_0(v85);
      if ( v84 > 0x40 && v83 )
        j_j___libc_free_0_0(v83);
      if ( v82 > 0x40 && v81 )
        j_j___libc_free_0_0(v81);
      if ( v72 > 0x40 && v71 )
        j_j___libc_free_0_0(v71);
      if ( v70 > 0x40 && v69 )
        j_j___libc_free_0_0(v69);
      if ( v62 > 0x40 && v61 )
        j_j___libc_free_0_0(v61);
      if ( v76 > 0x40 && v75 )
        j_j___libc_free_0_0(v75);
      if ( v74 > 0x40 && v73 )
        j_j___libc_free_0_0(v73);
      if ( v64 <= 0x40 )
        goto LABEL_5;
      v12 = v63;
      if ( !v63 )
        goto LABEL_5;
    }
  }
  j_j___libc_free_0_0(v12);
LABEL_5:
  if ( v68 > 0x40 && v67 )
    j_j___libc_free_0_0(v67);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  return a1;
}
