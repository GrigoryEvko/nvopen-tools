// Function: sub_11821B0
// Address: 0x11821b0
//
__int64 __fastcall sub_11821B0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 v6; // rbx
  __int64 v7; // r15
  _BYTE *v8; // r14
  __int64 v9; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  unsigned int v13; // r14d
  bool v14; // al
  int v15; // eax
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  unsigned __int8 *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rbx
  _QWORD *v24; // r14
  int v25; // eax
  __int64 v26; // rsi
  unsigned int v27; // eax
  char v28; // dl
  __int64 *v29; // r14
  unsigned __int8 *v30; // r14
  __int64 v31; // rcx
  unsigned __int8 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  unsigned __int8 *v35; // rsi
  unsigned __int8 *v36; // r15
  __int64 v37; // rbx
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // r14
  _BYTE *v41; // rax
  unsigned int v42; // r14d
  __int64 v43; // r14
  _BYTE *v44; // rax
  unsigned int v45; // r14d
  unsigned int v46; // eax
  char v47; // al
  bool v48; // r14
  __int64 v49; // rsi
  __int64 v50; // rax
  unsigned int v51; // r14d
  bool v52; // r14
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned int v55; // r14d
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // r15
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // rdi
  unsigned int v64; // eax
  _BYTE *v65; // rax
  unsigned __int8 *v66; // [rsp+8h] [rbp-178h]
  __int64 v67; // [rsp+10h] [rbp-170h]
  char v68; // [rsp+10h] [rbp-170h]
  int v69; // [rsp+18h] [rbp-168h]
  int v70; // [rsp+18h] [rbp-168h]
  _QWORD *v71; // [rsp+18h] [rbp-168h]
  char v72; // [rsp+20h] [rbp-160h]
  char v73; // [rsp+20h] [rbp-160h]
  char v74; // [rsp+20h] [rbp-160h]
  char v75; // [rsp+20h] [rbp-160h]
  char v76; // [rsp+20h] [rbp-160h]
  char v77; // [rsp+20h] [rbp-160h]
  __int64 v79; // [rsp+28h] [rbp-158h]
  __int64 v80; // [rsp+28h] [rbp-158h]
  unsigned int v81; // [rsp+30h] [rbp-150h]
  __int64 v82; // [rsp+30h] [rbp-150h]
  char v83; // [rsp+47h] [rbp-139h] BYREF
  __int64 v84; // [rsp+48h] [rbp-138h] BYREF
  _QWORD *v85; // [rsp+50h] [rbp-130h] BYREF
  unsigned __int8 *v86; // [rsp+58h] [rbp-128h] BYREF
  __int64 v87; // [rsp+60h] [rbp-120h] BYREF
  int v88; // [rsp+68h] [rbp-118h] BYREF
  char v89; // [rsp+6Ch] [rbp-114h]
  __int64 v90; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v91; // [rsp+78h] [rbp-108h]
  __int64 v92; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v93; // [rsp+88h] [rbp-F8h]
  _QWORD *v94; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v95; // [rsp+98h] [rbp-E8h] BYREF
  unsigned __int8 **v96; // [rsp+A0h] [rbp-E0h]
  _QWORD v97[4]; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v98; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v99; // [rsp+D8h] [rbp-A8h]
  __int64 *v100; // [rsp+E0h] [rbp-A0h]
  unsigned int v101; // [rsp+E8h] [rbp-98h]
  __int64 v102; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v103; // [rsp+F8h] [rbp-88h]
  __int64 v104; // [rsp+100h] [rbp-80h]
  unsigned int v105; // [rsp+108h] [rbp-78h]
  __int16 v106; // [rsp+110h] [rbp-70h]
  __int64 v107; // [rsp+120h] [rbp-60h] BYREF
  _QWORD *v108; // [rsp+128h] [rbp-58h]
  __int64 *v109; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v110; // [rsp+138h] [rbp-48h]
  __int16 v111; // [rsp+140h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_BCB060(v4);
  v6 = *(_QWORD *)(a1 - 32);
  v7 = *(_QWORD *)(a1 - 64);
  v88 = 42;
  v81 = v5;
  v8 = *(_BYTE **)(a1 - 96);
  v107 = (__int64)&v88;
  v108 = &v85;
  v89 = 0;
  v109 = &v84;
  LOBYTE(v110) = 0;
  if ( *v8 != 82 )
    return 0;
  if ( !*((_QWORD *)v8 - 8) )
    return 0;
  v85 = (_QWORD *)*((_QWORD *)v8 - 8);
  if ( !(unsigned __int8)sub_991580((__int64)&v109, *((_QWORD *)v8 - 4)) )
    return 0;
  if ( v107 )
  {
    v11 = sub_B53900((__int64)v8);
    v12 = v107;
    *(_DWORD *)v107 = v11;
    *(_BYTE *)(v12 + 4) = BYTE4(v11);
  }
  if ( *(_BYTE *)v7 == 17 )
  {
    v13 = *(_DWORD *)(v7 + 32);
    if ( v13 <= 0x40 )
      v14 = *(_QWORD *)(v7 + 24) == 1;
    else
      v14 = v13 - 1 == (unsigned int)sub_C444A0(v7 + 24);
    goto LABEL_10;
  }
  v43 = *(_QWORD *)(v7 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v43 + 8) - 17 > 1 || *(_BYTE *)v7 > 0x15u )
    goto LABEL_12;
  v44 = sub_AD7630(v7, 0, v9);
  if ( v44 && *v44 == 17 )
  {
    v45 = *((_DWORD *)v44 + 8);
    if ( v45 <= 0x40 )
      v14 = *((_QWORD *)v44 + 3) == 1;
    else
      v14 = v45 - 1 == (unsigned int)sub_C444A0((__int64)(v44 + 24));
LABEL_10:
    if ( v14 )
    {
LABEL_11:
      v15 = sub_B52870(v88);
      v89 = 0;
      v88 = v15;
      v16 = v6;
      v6 = v7;
      v7 = v16;
    }
    goto LABEL_12;
  }
  if ( *(_BYTE *)(v43 + 8) == 17 )
  {
    v69 = *(_DWORD *)(v43 + 32);
    if ( v69 )
    {
      v48 = 0;
      v49 = 0;
      while ( 1 )
      {
        v50 = sub_AD69F0((unsigned __int8 *)v7, v49);
        if ( !v50 )
          break;
        if ( *(_BYTE *)v50 != 13 )
        {
          if ( *(_BYTE *)v50 != 17 )
            break;
          v51 = *(_DWORD *)(v50 + 32);
          v48 = v51 <= 0x40 ? *(_QWORD *)(v50 + 24) == 1 : v51 - 1 == (unsigned int)sub_C444A0(v50 + 24);
          if ( !v48 )
            break;
        }
        v49 = (unsigned int)(v49 + 1);
        if ( v69 == (_DWORD)v49 )
        {
          if ( v48 )
            goto LABEL_11;
          break;
        }
      }
    }
  }
LABEL_12:
  if ( *(_BYTE *)v6 != 17 )
  {
    v40 = *(_QWORD *)(v6 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 > 1 || *(_BYTE *)v6 > 0x15u )
      return 0;
    v41 = sub_AD7630(v6, 0, v9);
    if ( !v41 || *v41 != 17 )
    {
      if ( *(_BYTE *)(v40 + 8) == 17 )
      {
        v70 = *(_DWORD *)(v40 + 32);
        if ( v70 )
        {
          v52 = 0;
          v53 = 0;
          while ( 1 )
          {
            v54 = sub_AD69F0((unsigned __int8 *)v6, v53);
            if ( !v54 )
              break;
            if ( *(_BYTE *)v54 != 13 )
            {
              if ( *(_BYTE *)v54 != 17 )
                break;
              v55 = *(_DWORD *)(v54 + 32);
              v52 = v55 <= 0x40 ? *(_QWORD *)(v54 + 24) == 1 : v55 - 1 == (unsigned int)sub_C444A0(v54 + 24);
              if ( !v52 )
                break;
            }
            v53 = (unsigned int)(v53 + 1);
            if ( v70 == (_DWORD)v53 )
            {
              if ( v52 )
                goto LABEL_15;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v42 = *((_DWORD *)v41 + 8);
    if ( v42 <= 0x40 )
    {
      if ( *((_QWORD *)v41 + 3) == 1 )
        goto LABEL_15;
    }
    else if ( (unsigned int)sub_C444A0((__int64)(v41 + 24)) == v42 - 1 )
    {
      goto LABEL_15;
    }
    return 0;
  }
  v17 = *(_DWORD *)(v6 + 32);
  if ( v17 <= 0x40 )
  {
    if ( *(_QWORD *)(v6 + 24) == 1 )
      goto LABEL_15;
    return 0;
  }
  if ( (unsigned int)sub_C444A0(v6 + 24) != v17 - 1 )
    return 0;
LABEL_15:
  v94 = 0;
  v95 = v81;
  v96 = &v86;
  v18 = *(_QWORD *)(v7 + 16);
  if ( !v18 )
    return 0;
  if ( *(_QWORD *)(v18 + 8) )
    return 0;
  if ( *(_BYTE *)v7 != 54 )
    return 0;
  if ( !(unsigned __int8)sub_993A50(&v94, *(_QWORD *)(v7 - 64)) )
    return 0;
  v19 = *(_QWORD *)(v7 - 32);
  v20 = *(_QWORD *)(v19 + 16);
  if ( !v20 )
    return 0;
  if ( *(_QWORD *)(v20 + 8) )
    return 0;
  if ( *(_BYTE *)v19 != 44 )
    return 0;
  if ( !sub_F17ED0(&v95, *(_QWORD *)(v19 - 64)) )
    return 0;
  v21 = *(unsigned __int8 **)(v19 - 32);
  if ( !v21 )
    return 0;
  *v96 = v21;
  if ( *v86 != 85 )
    return 0;
  v22 = *((_QWORD *)v86 - 4);
  if ( !v22 )
    return 0;
  if ( *(_BYTE *)v22 )
    return 0;
  if ( *(_QWORD *)(v22 + 24) != *((_QWORD *)v86 + 10) )
    return 0;
  if ( *(_DWORD *)(v22 + 36) != 65 )
    return 0;
  v66 = *(unsigned __int8 **)&v86[-32 * (*((_DWORD *)v86 + 1) & 0x7FFFFFF)];
  if ( !v66 )
    return 0;
  v23 = v84;
  v87 = *(_QWORD *)&v86[-32 * (*((_DWORD *)v86 + 1) & 0x7FFFFFF)];
  v24 = v85;
  v25 = sub_B52870(v88);
  sub_AB1A50((__int64)&v98, v25, v23);
  v26 = (__int64)v24;
  v97[0] = &v87;
  v83 = 0;
  v97[1] = &v83;
  v97[2] = &v98;
  v27 = sub_117F390((__int64)v97, v24);
  v28 = v27;
  if ( (_BYTE)v27 )
    goto LABEL_70;
  if ( *(_BYTE *)v24 == 42 )
  {
    v71 = (_QWORD *)*(v24 - 8);
    if ( v71 )
    {
      v63 = *(v24 - 4);
      v26 = v63 + 24;
      if ( *(_BYTE *)v63 != 17 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v63 + 8) + 8LL) - 17 > 1 )
          goto LABEL_32;
        if ( *(_BYTE *)v63 > 0x15u )
          goto LABEL_32;
        v26 = 0;
        v68 = v27;
        v65 = sub_AD7630(v63, 0, v27);
        v28 = v68;
        if ( !v65 || *v65 != 17 )
          goto LABEL_32;
        v26 = (__int64)(v65 + 24);
      }
      v93 = *(_DWORD *)(v26 + 8);
      if ( v93 > 0x40 )
        sub_C43780((__int64)&v92, (const void **)v26);
      else
        v92 = *(_QWORD *)v26;
      sub_AADBC0((__int64)&v102, &v92);
      sub_AB51C0((__int64)&v107, (__int64)&v98, (__int64)&v102);
      if ( v99 > 0x40 && v98 )
        j_j___libc_free_0_0(v98);
      v98 = v107;
      v64 = (unsigned int)v108;
      LODWORD(v108) = 0;
      v99 = v64;
      if ( v101 > 0x40 && v100 )
      {
        j_j___libc_free_0_0(v100);
        v100 = v109;
        v101 = v110;
        if ( (unsigned int)v108 > 0x40 && v107 )
          j_j___libc_free_0_0(v107);
      }
      else
      {
        v100 = v109;
        v101 = v110;
      }
      if ( v105 > 0x40 && v104 )
        j_j___libc_free_0_0(v104);
      if ( v103 > 0x40 && v102 )
        j_j___libc_free_0_0(v102);
      if ( v93 > 0x40 && v92 )
        j_j___libc_free_0_0(v92);
      v26 = (__int64)v71;
      v28 = sub_117F390((__int64)v97, v71);
      if ( v28 )
      {
LABEL_70:
        LODWORD(v108) = v81;
        if ( v81 > 0x40 )
        {
          v67 = 1LL << ((unsigned __int8)v81 - 1);
          sub_C43690((__int64)&v107, 0, 0);
          if ( (unsigned int)v108 <= 0x40 )
            v107 |= v67;
          else
            *(_QWORD *)(v107 + 8LL * ((v81 - 1) >> 6)) |= v67;
          sub_C46F20((__int64)&v107, 1u);
          v91 = (unsigned int)v108;
          v90 = v107;
          v93 = v81;
          sub_C43690((__int64)&v92, 1, 0);
        }
        else
        {
          v107 = 1LL << ((unsigned __int8)v81 - 1);
          sub_C46F20((__int64)&v107, 1u);
          v92 = 1;
          v91 = (unsigned int)v108;
          v90 = v107;
          v93 = v81;
        }
        sub_AADBC0((__int64)&v102, &v92);
        sub_AB51C0((__int64)&v107, (__int64)&v98, (__int64)&v102);
        if ( v99 > 0x40 && v98 )
          j_j___libc_free_0_0(v98);
        v98 = v107;
        v46 = (unsigned int)v108;
        LODWORD(v108) = 0;
        v99 = v46;
        if ( v101 > 0x40 && v100 )
        {
          j_j___libc_free_0_0(v100);
          v100 = v109;
          v101 = v110;
          if ( (unsigned int)v108 > 0x40 && v107 )
            j_j___libc_free_0_0(v107);
        }
        else
        {
          v100 = v109;
          v101 = v110;
        }
        if ( v105 > 0x40 && v104 )
          j_j___libc_free_0_0(v104);
        if ( v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        if ( v93 > 0x40 && v92 )
          j_j___libc_free_0_0(v92);
        v103 = v91;
        if ( v91 > 0x40 )
          sub_C43780((__int64)&v102, (const void **)&v90);
        else
          v102 = v90;
        sub_AADBC0((__int64)&v107, &v102);
        v26 = 35;
        v47 = sub_ABB410(&v98, 35, &v107);
        v28 = v47;
        if ( v110 > 0x40 && v109 )
        {
          v74 = v47;
          j_j___libc_free_0_0(v109);
          v28 = v74;
        }
        if ( (unsigned int)v108 > 0x40 && v107 )
        {
          v75 = v28;
          j_j___libc_free_0_0(v107);
          v28 = v75;
        }
        if ( v103 > 0x40 && v102 )
        {
          v76 = v28;
          j_j___libc_free_0_0(v102);
          v28 = v76;
        }
        if ( v91 > 0x40 && v90 )
        {
          v77 = v28;
          j_j___libc_free_0_0(v90);
          v28 = v77;
        }
      }
    }
  }
LABEL_32:
  if ( v101 > 0x40 && v100 )
  {
    v72 = v28;
    j_j___libc_free_0_0(v100);
    v28 = v72;
  }
  if ( v99 > 0x40 && v98 )
  {
    v73 = v28;
    j_j___libc_free_0_0(v98);
    v28 = v73;
  }
  if ( !v28 )
    return 0;
  if ( v83 )
  {
    sub_B447F0(v66, 0);
    v26 = 0;
    sub_B44850(v66, 0);
  }
  v29 = (__int64 *)v86;
  sub_B44F30(v86);
  sub_B44B50(v29, v26);
  sub_B44A60((__int64)v29);
  v30 = v86;
  v31 = sub_ACD720((__int64 *)a2[9]);
  if ( (v30[7] & 0x40) != 0 )
    v32 = (unsigned __int8 *)*((_QWORD *)v30 - 1);
  else
    v32 = &v30[-32 * (*((_DWORD *)v30 + 1) & 0x7FFFFFF)];
  if ( *((_QWORD *)v32 + 4) )
  {
    v33 = *((_QWORD *)v32 + 5);
    **((_QWORD **)v32 + 6) = v33;
    if ( v33 )
      *(_QWORD *)(v33 + 16) = *((_QWORD *)v32 + 6);
  }
  *((_QWORD *)v32 + 4) = v31;
  if ( v31 )
  {
    v34 = *(_QWORD *)(v31 + 16);
    *((_QWORD *)v32 + 5) = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 16) = v32 + 40;
    *((_QWORD *)v32 + 6) = v31 + 16;
    *(_QWORD *)(v31 + 16) = v32 + 32;
  }
  v35 = v86;
  sub_F15FC0(*(_QWORD *)(a3 + 40), (__int64)v86);
  v36 = v86;
  v106 = 257;
  v79 = sub_AD6530(*((_QWORD *)v86 + 1), (__int64)v35);
  v37 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, _QWORD, _QWORD))(*(_QWORD *)a2[10]
                                                                                                + 32LL))(
          a2[10],
          15,
          v79,
          v36,
          0,
          0);
  if ( !v37 )
  {
    v111 = 257;
    v37 = sub_B504D0(15, v79, (__int64)v36, (__int64)&v107, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v37,
      &v102,
      a2[7],
      a2[8]);
    v60 = *a2;
    v80 = *a2 + 16LL * *((unsigned int *)a2 + 2);
    if ( *a2 != v80 )
    {
      do
      {
        v61 = *(_QWORD *)(v60 + 8);
        v62 = *(_DWORD *)v60;
        v60 += 16;
        sub_B99FD0(v37, v62, v61);
      }
      while ( v80 != v60 );
    }
  }
  v106 = 257;
  v82 = sub_AD64C0(v4, v81 - 1, 0);
  v38 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a2[10] + 16LL))(a2[10], 28, v37, v82);
  if ( !v38 )
  {
    v111 = 257;
    v38 = sub_B504D0(28, v37, v82, (__int64)&v107, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
      a2[11],
      v38,
      &v102,
      a2[7],
      a2[8]);
    v56 = *a2;
    v57 = *a2 + 16LL * *((unsigned int *)a2 + 2);
    while ( v57 != v56 )
    {
      v58 = *(_QWORD *)(v56 + 8);
      v59 = *(_DWORD *)v56;
      v56 += 16;
      sub_B99FD0(v38, v59, v58);
    }
  }
  v111 = 257;
  v39 = sub_AD64C0(v4, 1, 0);
  return sub_B504D0(25, v39, v38, (__int64)&v107, 0, 0);
}
