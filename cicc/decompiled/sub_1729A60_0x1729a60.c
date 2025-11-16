// Function: sub_1729A60
// Address: 0x1729a60
//
__int64 __fastcall sub_1729A60(
        __int64 *a1,
        __int64 a2,
        unsigned __int8 a3,
        unsigned __int8 *a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        __int64 a10,
        int a11,
        __int64 a12)
{
  __int64 v14; // r13
  unsigned int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // r15
  _QWORD *v18; // r15
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // r15
  unsigned int v24; // eax
  __int64 v25; // rcx
  unsigned int v26; // edx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // r12
  unsigned __int8 *v35; // rax
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rdi
  char v39; // r15
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // r15
  bool v44; // al
  unsigned int v45; // r15d
  bool v46; // dl
  unsigned int v47; // eax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r12
  bool v51; // bl
  unsigned int v52; // eax
  unsigned int v53; // edx
  unsigned int v54; // eax
  __int64 v55; // r8
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r8
  int v59; // eax
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // r15
  bool v63; // r12
  __int64 v64; // r15
  __int64 v65; // rax
  __int64 v66; // r15
  bool v67; // al
  __int64 v68; // rax
  __int64 v69; // rax
  _QWORD *v70; // r13
  unsigned int v71; // eax
  unsigned int v72; // eax
  int v73; // eax
  unsigned int v74; // eax
  unsigned int v75; // eax
  unsigned int v76; // ebx
  __int64 v77; // [rsp+0h] [rbp-F0h]
  unsigned int v78; // [rsp+Ch] [rbp-E4h]
  unsigned int v79; // [rsp+Ch] [rbp-E4h]
  unsigned int v80; // [rsp+Ch] [rbp-E4h]
  __int64 *v81; // [rsp+10h] [rbp-E0h]
  int v85; // [rsp+34h] [rbp-BCh]
  int v86; // [rsp+38h] [rbp-B8h]
  bool v87; // [rsp+38h] [rbp-B8h]
  unsigned int v88; // [rsp+38h] [rbp-B8h]
  bool v89; // [rsp+40h] [rbp-B0h]
  bool v90; // [rsp+40h] [rbp-B0h]
  __int64 *v91; // [rsp+48h] [rbp-A8h]
  __int64 v92; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v93; // [rsp+58h] [rbp-98h]
  __int64 v94; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v95; // [rsp+68h] [rbp-88h]
  __int64 v96; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v97; // [rsp+78h] [rbp-78h]
  _QWORD *v98; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v99; // [rsp+88h] [rbp-68h]
  _QWORD *v100; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v101; // [rsp+98h] [rbp-58h]
  _QWORD *v102; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v103; // [rsp+A8h] [rbp-48h]
  __int16 v104; // [rsp+B0h] [rbp-40h]

  v14 = a10;
  v85 = (a3 == 0) + 32;
  if ( v85 != a11 )
    v14 = sub_15A2D30((__int64 *)a6, a10, a7, a8, a9);
  v15 = *(_DWORD *)(a5 + 32);
  if ( v15 > 0x40 )
  {
    if ( v15 - (unsigned int)sub_16A57B0(a5 + 24) <= 0x40 && !**(_QWORD **)(a5 + 24) )
      return 0;
  }
  else if ( !*(_QWORD *)(a5 + 24) )
  {
    return 0;
  }
  v91 = (__int64 *)(a6 + 24);
  if ( *(_DWORD *)(a6 + 32) <= 0x40u )
  {
    v16 = *(_QWORD *)(a6 + 24);
  }
  else
  {
    v86 = *(_DWORD *)(a6 + 32);
    if ( v86 - (unsigned int)sub_16A57B0(a6 + 24) > 0x40 )
      goto LABEL_9;
    v16 = **(_QWORD **)(a6 + 24);
  }
  if ( !v16 )
    return 0;
LABEL_9:
  v101 = v15;
  if ( v15 <= 0x40 )
  {
    v17 = *(_QWORD *)(a5 + 24);
LABEL_11:
    v18 = (_QWORD *)(*(_QWORD *)(a6 + 24) & v17);
    v101 = 0;
    v100 = v18;
    goto LABEL_12;
  }
  sub_16A4FD0((__int64)&v100, (const void **)(a5 + 24));
  if ( v101 <= 0x40 )
  {
    v17 = (__int64)v100;
    goto LABEL_11;
  }
  sub_16A8890((__int64 *)&v100, v91);
  v53 = v101;
  v18 = v100;
  v101 = 0;
  v103 = v53;
  v102 = v100;
  if ( v53 <= 0x40 )
  {
LABEL_12:
    if ( v18 )
      goto LABEL_13;
    return 0;
  }
  if ( v53 - (unsigned int)sub_16A57B0((__int64)&v102) > 0x40 || *v18 )
  {
    if ( !v18 )
      goto LABEL_13;
    j_j___libc_free_0_0(v18);
    if ( v101 <= 0x40 )
      goto LABEL_13;
    v38 = (__int64)v100;
    v39 = 0;
    if ( !v100 )
      goto LABEL_13;
  }
  else
  {
    j_j___libc_free_0_0(v18);
    if ( v101 <= 0x40 )
      return 0;
    v38 = (__int64)v100;
    v39 = 1;
    if ( !v100 )
      return 0;
  }
  j_j___libc_free_0_0(v38);
  if ( v39 )
    return 0;
LABEL_13:
  v19 = *(_DWORD *)(a5 + 32);
  v81 = (__int64 *)(v14 + 24);
  v99 = v19;
  if ( v19 <= 0x40 )
  {
    v20 = *(_QWORD *)(a5 + 24);
LABEL_15:
    v21 = *(_QWORD *)(a6 + 24) & v20;
    v99 = 0;
    v98 = (_QWORD *)v21;
LABEL_16:
    v22 = *(_QWORD *)(v14 + 24) & v21;
    v103 = v19;
    v100 = (_QWORD *)v22;
    v23 = (__int64 *)v22;
    v102 = (_QWORD *)v22;
    v101 = 0;
    goto LABEL_17;
  }
  sub_16A4FD0((__int64)&v98, (const void **)(a5 + 24));
  v19 = v99;
  if ( v99 <= 0x40 )
  {
    v20 = (__int64)v98;
    goto LABEL_15;
  }
  sub_16A8890((__int64 *)&v98, v91);
  v19 = v99;
  v21 = (__int64)v98;
  v99 = 0;
  v101 = v19;
  v100 = v98;
  if ( v19 <= 0x40 )
    goto LABEL_16;
  sub_16A8890((__int64 *)&v100, v81);
  v19 = v101;
  v23 = v100;
  v101 = 0;
  v103 = v19;
  v102 = v100;
  if ( v19 <= 0x40 )
  {
    v22 = (__int64)v100;
  }
  else
  {
    v88 = v19;
    v59 = sub_16A57B0((__int64)&v102);
    v19 = v88;
    if ( v88 - v59 > 0x40 )
      goto LABEL_18;
    v22 = *v23;
  }
LABEL_17:
  if ( v22 )
  {
LABEL_18:
    v87 = 0;
    goto LABEL_19;
  }
  v54 = *(_DWORD *)(a5 + 32);
  v93 = v54;
  if ( v54 <= 0x40 )
  {
    v55 = *(_QWORD *)(a5 + 24);
    v56 = v55;
LABEL_91:
    v93 = 0;
    v57 = *(_QWORD *)(a6 + 24) ^ v55;
    v92 = v57;
LABEL_92:
    v58 = v56 & v57;
    v97 = v54;
    v94 = v58;
    v96 = v58;
    v95 = 0;
    goto LABEL_93;
  }
  v80 = v19;
  sub_16A4FD0((__int64)&v92, (const void **)(a5 + 24));
  v54 = v93;
  v19 = v80;
  if ( v93 <= 0x40 )
  {
    v55 = v92;
    v56 = *(_QWORD *)(a5 + 24);
    goto LABEL_91;
  }
  sub_16A8F00(&v92, v91);
  v54 = v93;
  v57 = v92;
  v93 = 0;
  v19 = v80;
  v95 = v54;
  v94 = v92;
  if ( v54 <= 0x40 )
  {
    v56 = *(_QWORD *)(a5 + 24);
    goto LABEL_92;
  }
  sub_16A8890(&v94, (__int64 *)(a5 + 24));
  v72 = v95;
  v58 = v94;
  v95 = 0;
  v19 = v80;
  v97 = v72;
  v96 = v94;
  if ( v72 > 0x40 )
  {
    v77 = v94;
    v73 = sub_16A5940((__int64)&v96);
    v19 = v80;
    v87 = v73 == 1;
    if ( v77 )
    {
      j_j___libc_free_0_0(v77);
      v19 = v80;
    }
    goto LABEL_95;
  }
LABEL_93:
  v87 = 0;
  if ( v58 )
    v87 = (v58 & (v58 - 1)) == 0;
LABEL_95:
  if ( v95 > 0x40 && v94 )
  {
    v78 = v19;
    j_j___libc_free_0_0(v94);
    v19 = v78;
  }
  if ( v93 > 0x40 && v92 )
  {
    v79 = v19;
    j_j___libc_free_0_0(v92);
    v19 = v79;
  }
LABEL_19:
  if ( v19 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v101 > 0x40 && v100 )
    j_j___libc_free_0_0(v100);
  if ( v99 > 0x40 && v98 )
    j_j___libc_free_0_0(v98);
  v24 = *(_DWORD *)(a5 + 32);
  v103 = v24;
  if ( v87 )
  {
    if ( v24 > 0x40 )
    {
      sub_16A4FD0((__int64)&v102, (const void **)(a5 + 24));
      v24 = v103;
      if ( v103 > 0x40 )
      {
        sub_16A89F0((__int64 *)&v102, v91);
        v24 = v103;
        v27 = (__int64)v102;
        v26 = *(_DWORD *)(a5 + 32);
LABEL_32:
        v95 = v24;
        v94 = v27;
        v99 = v26;
        if ( v26 > 0x40 )
        {
          sub_16A4FD0((__int64)&v98, (const void **)(a5 + 24));
          v26 = v99;
          if ( v99 > 0x40 )
          {
            sub_16A8F00((__int64 *)&v98, v91);
            v26 = v99;
            v30 = (__int64)v98;
            v99 = 0;
            v101 = v26;
            v100 = v98;
            if ( v26 > 0x40 )
            {
              sub_16A8890((__int64 *)&v100, (__int64 *)(a5 + 24));
              v26 = v101;
              v31 = (__int64)v100;
              v101 = 0;
              v103 = v26;
              v102 = v100;
              if ( v26 > 0x40 )
              {
                sub_16A89F0((__int64 *)&v102, v81);
                v97 = v103;
                v96 = (__int64)v102;
                if ( v101 > 0x40 && v100 )
                  j_j___libc_free_0_0(v100);
                goto LABEL_37;
              }
LABEL_36:
              v32 = *(_QWORD *)(v14 + 24) | v31;
              v97 = v26;
              v96 = v32;
LABEL_37:
              if ( v99 > 0x40 && v98 )
                j_j___libc_free_0_0(v98);
              v33 = sub_15A1070(*(_QWORD *)a5, (__int64)&v94);
              v34 = sub_15A1070(*(_QWORD *)a5, (__int64)&v96);
              v104 = 257;
              v35 = sub_1729500(a12, a4, v33, (__int64 *)&v102, a7, a8, a9);
              v104 = 257;
              if ( v35[16] > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
              {
                v36 = (__int64)sub_1727440(a12, v85, (__int64)v35, v34, (__int64 *)&v102);
              }
              else
              {
                v36 = sub_15A37B0(v85, v35, (_QWORD *)v34, 0);
                v37 = sub_14DBA30(v36, *(_QWORD *)(a12 + 96), 0);
                if ( v37 )
                  v36 = v37;
              }
              if ( v97 > 0x40 && v96 )
                j_j___libc_free_0_0(v96);
              if ( v95 > 0x40 )
              {
                if ( v94 )
                  j_j___libc_free_0_0(v94);
              }
              return v36;
            }
            v29 = *(_QWORD *)(a5 + 24);
LABEL_35:
            v31 = v29 & v30;
            goto LABEL_36;
          }
          v28 = (__int64)v98;
          v29 = *(_QWORD *)(a5 + 24);
        }
        else
        {
          v28 = *(_QWORD *)(a5 + 24);
          v29 = v28;
        }
        v30 = *(_QWORD *)(a6 + 24) ^ v28;
        v99 = 0;
        v98 = (_QWORD *)v30;
        goto LABEL_35;
      }
      v25 = (__int64)v102;
      v26 = *(_DWORD *)(a5 + 32);
    }
    else
    {
      v25 = *(_QWORD *)(a5 + 24);
      v26 = v24;
    }
    v27 = *(_QWORD *)(a6 + 24) | v25;
    goto LABEL_32;
  }
  if ( v24 <= 0x40 )
  {
    v41 = *(_QWORD *)(a5 + 24);
    v42 = v41;
LABEL_62:
    v43 = *(_QWORD *)(a6 + 24) & v41;
LABEL_63:
    v44 = v43 == v42;
    goto LABEL_64;
  }
  sub_16A4FD0((__int64)&v102, (const void **)(a5 + 24));
  if ( v103 <= 0x40 )
  {
    v41 = (__int64)v102;
    v42 = *(_QWORD *)(a5 + 24);
    goto LABEL_62;
  }
  sub_16A8890((__int64 *)&v102, v91);
  v52 = v103;
  v43 = (__int64)v102;
  v103 = 0;
  v101 = v52;
  v100 = v102;
  if ( v52 <= 0x40 )
  {
    v42 = *(_QWORD *)(a5 + 24);
    goto LABEL_63;
  }
  v44 = sub_16A5220((__int64)&v100, (const void **)(a5 + 24));
  if ( v43 )
  {
    v89 = v44;
    j_j___libc_free_0_0(v43);
    v44 = v89;
    if ( v103 > 0x40 )
    {
      if ( v102 )
      {
        j_j___libc_free_0_0(v102);
        v44 = v89;
      }
    }
  }
LABEL_64:
  if ( v44 )
    goto LABEL_65;
  v103 = *(_DWORD *)(a5 + 32);
  if ( v103 <= 0x40 )
  {
    v64 = *(_QWORD *)(a5 + 24);
LABEL_121:
    v65 = *(_QWORD *)(a6 + 24);
    v66 = v65 & v64;
LABEL_122:
    v67 = v66 == v65;
    goto LABEL_123;
  }
  sub_16A4FD0((__int64)&v102, (const void **)(a5 + 24));
  if ( v103 <= 0x40 )
  {
    v64 = (__int64)v102;
    goto LABEL_121;
  }
  sub_16A8890((__int64 *)&v102, v91);
  v71 = v103;
  v66 = (__int64)v102;
  v103 = 0;
  v101 = v71;
  v100 = v102;
  if ( v71 <= 0x40 )
  {
    v65 = *(_QWORD *)(a6 + 24);
    goto LABEL_122;
  }
  v67 = sub_16A5220((__int64)&v100, (const void **)v91);
  if ( v66 )
  {
    v90 = v67;
    j_j___libc_free_0_0(v66);
    v67 = v90;
    if ( v103 > 0x40 )
    {
      if ( v102 )
      {
        j_j___libc_free_0_0(v102);
        v67 = v90;
      }
    }
  }
LABEL_123:
  if ( !v67 )
    return 0;
LABEL_65:
  v45 = *(_DWORD *)(v14 + 32);
  if ( v45 <= 0x40 )
    v46 = *(_QWORD *)(v14 + 24) == 0;
  else
    v46 = v45 == (unsigned int)sub_16A57B0((__int64)v81);
  v47 = *(_DWORD *)(a5 + 32);
  if ( !v46 )
  {
    v103 = *(_DWORD *)(a5 + 32);
    if ( v47 > 0x40 )
    {
      sub_16A4FD0((__int64)&v102, (const void **)(a5 + 24));
      if ( v103 > 0x40 )
      {
        sub_16A8890((__int64 *)&v102, v91);
        v74 = v103;
        v62 = (__int64)v102;
        v103 = 0;
        v101 = v74;
        v100 = v102;
        if ( v74 > 0x40 )
        {
          v63 = sub_16A5220((__int64)&v100, (const void **)v91);
          if ( v62 )
          {
            j_j___libc_free_0_0(v62);
            if ( v103 > 0x40 )
            {
              if ( v102 )
                j_j___libc_free_0_0(v102);
            }
          }
LABEL_111:
          if ( v63 )
            return a2;
          v101 = *(_DWORD *)(a5 + 32);
          if ( v101 <= 0x40 )
          {
            v68 = *(_QWORD *)(a5 + 24);
LABEL_128:
            v69 = *(_QWORD *)(v14 + 24) & v68;
            v101 = 0;
            v100 = (_QWORD *)v69;
            v70 = (_QWORD *)v69;
            goto LABEL_129;
          }
          sub_16A4FD0((__int64)&v100, (const void **)(a5 + 24));
          if ( v101 <= 0x40 )
          {
            v68 = (__int64)v100;
            goto LABEL_128;
          }
          sub_16A8890((__int64 *)&v100, v81);
          v76 = v101;
          v70 = v100;
          v101 = 0;
          v103 = v76;
          v102 = v100;
          if ( v76 <= 0x40 )
          {
LABEL_129:
            if ( !v70 )
              return sub_15A0680(*a1, a3 ^ 1u, 0);
            return a2;
          }
          if ( v76 - (unsigned int)sub_16A57B0((__int64)&v102) <= 0x40 )
          {
            if ( *v70 )
            {
              j_j___libc_free_0_0(v70);
              if ( v101 <= 0x40 )
                return a2;
LABEL_163:
              if ( v100 )
                j_j___libc_free_0_0(v100);
LABEL_158:
              if ( v63 )
                return sub_15A0680(*a1, a3 ^ 1u, 0);
              return a2;
            }
            v63 = 1;
          }
          if ( !v70 )
            goto LABEL_158;
          j_j___libc_free_0_0(v70);
          if ( v101 <= 0x40 )
            goto LABEL_158;
          goto LABEL_163;
        }
        v61 = *(_QWORD *)(a6 + 24);
LABEL_110:
        v63 = v62 == v61;
        goto LABEL_111;
      }
      v60 = (__int64)v102;
    }
    else
    {
      v60 = *(_QWORD *)(a5 + 24);
    }
    v61 = *(_QWORD *)(a6 + 24);
    v62 = v61 & v60;
    goto LABEL_110;
  }
  v103 = *(_DWORD *)(a5 + 32);
  if ( v47 <= 0x40 )
  {
    v48 = *(_QWORD *)(a5 + 24);
    v49 = v48;
LABEL_70:
    v50 = *(_QWORD *)(a6 + 24) & v48;
LABEL_71:
    v51 = v50 == v49;
    goto LABEL_72;
  }
  sub_16A4FD0((__int64)&v102, (const void **)(a5 + 24));
  if ( v103 <= 0x40 )
  {
    v48 = (__int64)v102;
    v49 = *(_QWORD *)(a5 + 24);
    goto LABEL_70;
  }
  sub_16A8890((__int64 *)&v102, v91);
  v75 = v103;
  v50 = (__int64)v102;
  v103 = 0;
  v101 = v75;
  v100 = v102;
  if ( v75 <= 0x40 )
  {
    v49 = *(_QWORD *)(a5 + 24);
    goto LABEL_71;
  }
  v51 = sub_16A5220((__int64)&v100, (const void **)(a5 + 24));
  if ( v50 )
  {
    j_j___libc_free_0_0(v50);
    if ( v103 > 0x40 )
    {
      if ( v102 )
        j_j___libc_free_0_0(v102);
    }
  }
LABEL_72:
  if ( !v51 )
    return 0;
  return sub_15A0680(*a1, a3 ^ 1u, 0);
}
