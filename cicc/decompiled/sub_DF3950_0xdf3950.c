// Function: sub_DF3950
// Address: 0xdf3950
//
__int64 __fastcall sub_DF3950(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 result; // rax
  __int64 *v10; // r14
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  unsigned int v44; // eax
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rsi
  bool v48; // zf
  unsigned __int64 v49; // rax
  unsigned int v50; // eax
  __int64 v51; // rdi
  __int64 v52; // rax
  unsigned int v53; // eax
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rsi
  unsigned __int64 v57; // rax
  unsigned int v58; // eax
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // [rsp+8h] [rbp-118h]
  unsigned __int64 v62; // [rsp+8h] [rbp-118h]
  __int64 v63; // [rsp+8h] [rbp-118h]
  unsigned __int64 v64; // [rsp+8h] [rbp-118h]
  unsigned int v65; // [rsp+10h] [rbp-110h]
  unsigned int v66; // [rsp+10h] [rbp-110h]
  __int64 v67; // [rsp+18h] [rbp-108h]
  __int64 v68; // [rsp+18h] [rbp-108h]
  __int64 *v69; // [rsp+18h] [rbp-108h]
  _QWORD *v70; // [rsp+18h] [rbp-108h]
  _QWORD *v71; // [rsp+18h] [rbp-108h]
  __int64 v72; // [rsp+18h] [rbp-108h]
  __int64 v73; // [rsp+18h] [rbp-108h]
  __int64 *v74; // [rsp+20h] [rbp-100h]
  __int64 v75; // [rsp+20h] [rbp-100h]
  __int64 v76; // [rsp+20h] [rbp-100h]
  __int64 *v77; // [rsp+20h] [rbp-100h]
  bool v78; // [rsp+20h] [rbp-100h]
  bool v79; // [rsp+20h] [rbp-100h]
  unsigned int v80; // [rsp+20h] [rbp-100h]
  unsigned int v81; // [rsp+20h] [rbp-100h]
  unsigned int v82; // [rsp+20h] [rbp-100h]
  unsigned __int64 v84; // [rsp+30h] [rbp-F0h] BYREF
  unsigned int v85; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v86; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v87; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v88; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v89; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v90; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v91; // [rsp+68h] [rbp-B8h]
  unsigned __int64 v92; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v93; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v94; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v95; // [rsp+88h] [rbp-98h]
  __int64 v96; // [rsp+90h] [rbp-90h] BYREF
  __int64 v97; // [rsp+98h] [rbp-88h]
  __int64 v98; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-78h]
  __int64 v100; // [rsp+B0h] [rbp-70h]
  __int64 v101; // [rsp+B8h] [rbp-68h]
  __int64 v102; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v103; // [rsp+C8h] [rbp-58h]
  __int64 v104; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-48h]
  __int64 v106; // [rsp+E0h] [rbp-40h]
  __int64 v107; // [rsp+E8h] [rbp-38h]

  v7 = a2[1];
  if ( v7 != -1 && v7 != 0xBFFFFFFFFFFFFFFELL && (v7 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
    return 0;
  v8 = a3[1];
  if ( v8 != -1 && v8 != 0xBFFFFFFFFFFFFFFELL && (v8 & 0x3FFFFFFFFFFFFFFFLL) == 0 )
    return 0;
  v10 = sub_DD8400(*a1, *a2);
  v11 = sub_DD8400(*a1, *a3);
  result = 3;
  if ( v10 != v11 )
  {
    v74 = (__int64 *)*a1;
    v12 = sub_D95540((__int64)v10);
    v67 = sub_D97090((__int64)v74, v12);
    v13 = sub_D95540((__int64)v11);
    if ( v67 != sub_D97090((__int64)v74, v13)
      || !(unsigned __int8)sub_D97920(v74, (__int64)v10, (__int64)v11, v14, v15, v16) )
    {
      goto LABEL_11;
    }
    v75 = *a1;
    v29 = sub_D95540((__int64)v10);
    v30 = sub_D97050(v75, v29);
    v31 = a2[1];
    if ( v31 == 0xBFFFFFFFFFFFFFFELL || v31 == -1 )
    {
      v32 = -1;
    }
    else
    {
      v82 = v30;
      v102 = v31 & 0x3FFFFFFFFFFFFFFFLL;
      LOBYTE(v103) = (v31 & 0x4000000000000000LL) != 0;
      v43 = sub_CA1930(&v102);
      v30 = v82;
      v32 = v43;
    }
    v85 = v30;
    if ( v30 > 0x40 )
    {
      v81 = v30;
      sub_C43690((__int64)&v84, v32, 0);
      v41 = a3[1];
      if ( v41 == 0xBFFFFFFFFFFFFFFELL || v41 == -1 )
      {
        v87 = v81;
        v42 = -1;
      }
      else
      {
        v102 = v41 & 0x3FFFFFFFFFFFFFFFLL;
        LOBYTE(v103) = (v41 & 0x4000000000000000LL) != 0;
        v42 = sub_CA1930(&v102);
        v87 = v81;
      }
      sub_C43690((__int64)&v86, v42, 0);
    }
    else
    {
      v33 = a3[1];
      v84 = v32;
      if ( v33 == -1 || v33 == 0xBFFFFFFFFFFFFFFELL )
      {
        v87 = v30;
        v34 = -1;
      }
      else
      {
        v80 = v30;
        v102 = v33 & 0x3FFFFFFFFFFFFFFFLL;
        LOBYTE(v103) = (v33 & 0x4000000000000000LL) != 0;
        v34 = sub_CA1930(&v102);
        v87 = v80;
      }
      v86 = v34;
    }
    v76 = *a1;
    v35 = sub_D95540((__int64)v10);
    v36 = sub_D97090(v76, v35);
    v77 = sub_DD3A70(v76, (__int64)v10, v36);
    v68 = *a1;
    v37 = sub_D95540((__int64)v11);
    v38 = sub_D97090(v68, v37);
    v69 = sub_DD3A70(v68, (__int64)v11, v38);
    if ( !sub_D96A50((__int64)v77) && !sub_D96A50((__int64)v69) )
    {
      v11 = v69;
      v10 = v77;
    }
    v70 = sub_DCC810((__int64 *)*a1, (__int64)v11, (__int64)v10, 0, 0);
    v78 = sub_D96A50((__int64)v70);
    if ( v78 )
      goto LABEL_34;
    v40 = sub_DBB9F0(*a1, (__int64)v70, 0, 0);
    LODWORD(v97) = *(_DWORD *)(v40 + 8);
    if ( (unsigned int)v97 > 0x40 )
    {
      v61 = v40;
      sub_C43780((__int64)&v96, (const void **)v40);
      v40 = v61;
    }
    else
    {
      v96 = *(_QWORD *)v40;
    }
    LODWORD(v99) = *(_DWORD *)(v40 + 24);
    if ( (unsigned int)v99 > 0x40 )
      sub_C43780((__int64)&v98, (const void **)(v40 + 16));
    else
      v98 = *(_QWORD *)(v40 + 16);
    sub_AB0A00((__int64)&v88, (__int64)&v96);
    if ( (int)sub_C49970((__int64)&v84, &v88) > 0 )
    {
LABEL_72:
      if ( v89 > 0x40 && v88 )
        j_j___libc_free_0_0(v88);
      if ( (unsigned int)v99 > 0x40 && v98 )
        j_j___libc_free_0_0(v98);
      if ( (unsigned int)v97 > 0x40 && v96 )
        j_j___libc_free_0_0(v96);
      if ( v78 )
      {
LABEL_58:
        result = 0;
        if ( v87 > 0x40 && v86 )
        {
          j_j___libc_free_0_0(v86);
          result = 0;
        }
        if ( v85 > 0x40 && v84 )
        {
          j_j___libc_free_0_0(v84);
          return 0;
        }
        return result;
      }
LABEL_34:
      v71 = sub_DCC810((__int64 *)*a1, (__int64)v10, (__int64)v11, 0, 0);
      v79 = sub_D96A50((__int64)v71);
      if ( v79 )
      {
LABEL_35:
        if ( v87 > 0x40 && v86 )
          j_j___libc_free_0_0(v86);
        if ( v85 > 0x40 && v84 )
          j_j___libc_free_0_0(v84);
LABEL_11:
        v17 = sub_DF38E0((__int64)a1, (__int64)v10);
        v18 = sub_DF38E0((__int64)a1, (__int64)v11);
        if ( !v17 || *a2 == v17 )
        {
          if ( !v18 || *a3 == v18 )
            return 1;
          v102 = v18;
          v103 = -1;
          v104 = 0;
          v105 = 0;
          v106 = 0;
          v107 = 0;
          if ( !v17 )
          {
            v25 = a2[2];
            v24 = a2[3];
            v28 = a2[4];
            v27 = a2[5];
            v26 = a2[1];
            v17 = *a2;
LABEL_17:
            v97 = v26;
            v98 = v25;
            v99 = v24;
            v101 = v27;
            v96 = v17;
            v100 = v28;
            return (unsigned __int8)sub_DF3950(a1, &v96, &v102, a4, 0) != 0;
          }
        }
        else
        {
          if ( v18 )
          {
            v19 = 0;
            v20 = 0;
            v21 = 0;
            v22 = 0;
            v23 = -1;
          }
          else
          {
            v19 = a3[2];
            v20 = a3[3];
            v21 = a3[4];
            v22 = a3[5];
            v23 = a3[1];
            v18 = *a3;
          }
          v102 = v18;
          v103 = v23;
          v104 = v19;
          v105 = v20;
          v106 = v21;
          v107 = v22;
        }
        v24 = 0;
        v25 = 0;
        v26 = -1;
        v27 = 0;
        v28 = 0;
        goto LABEL_17;
      }
      v39 = sub_DBB9F0(*a1, (__int64)v71, 0, 0);
      LODWORD(v97) = *(_DWORD *)(v39 + 8);
      if ( (unsigned int)v97 > 0x40 )
      {
        v63 = v39;
        sub_C43780((__int64)&v96, (const void **)v39);
        v39 = v63;
      }
      else
      {
        v96 = *(_QWORD *)v39;
      }
      LODWORD(v99) = *(_DWORD *)(v39 + 24);
      if ( (unsigned int)v99 > 0x40 )
        sub_C43780((__int64)&v98, (const void **)(v39 + 16));
      else
        v98 = *(_QWORD *)(v39 + 16);
      sub_AB0A00((__int64)&v88, (__int64)&v96);
      if ( (int)sub_C49970((__int64)&v86, &v88) > 0 )
        goto LABEL_48;
      v53 = v85;
      v91 = v85;
      if ( v85 > 0x40 )
      {
        sub_C43780((__int64)&v90, (const void **)&v84);
        v53 = v91;
        if ( v91 > 0x40 )
        {
          sub_C43D10((__int64)&v90);
LABEL_124:
          sub_C46250((__int64)&v90);
          v58 = v91;
          v59 = *a1;
          v91 = 0;
          v66 = v58;
          v93 = v58;
          v64 = v90;
          v92 = v90;
          v60 = sub_DBB9F0(v59, (__int64)v71, 0, 0);
          LODWORD(v103) = *(_DWORD *)(v60 + 8);
          if ( (unsigned int)v103 > 0x40 )
          {
            v73 = v60;
            sub_C43780((__int64)&v102, (const void **)v60);
            v60 = v73;
          }
          else
          {
            v102 = *(_QWORD *)v60;
          }
          LODWORD(v105) = *(_DWORD *)(v60 + 24);
          if ( (unsigned int)v105 > 0x40 )
            sub_C43780((__int64)&v104, (const void **)(v60 + 16));
          else
            v104 = *(_QWORD *)(v60 + 16);
          sub_AB0910((__int64)&v94, (__int64)&v102);
          v79 = (int)sub_C49970((__int64)&v92, &v94) >= 0;
          if ( v95 > 0x40 && v94 )
            j_j___libc_free_0_0(v94);
          if ( (unsigned int)v105 > 0x40 && v104 )
            j_j___libc_free_0_0(v104);
          if ( (unsigned int)v103 > 0x40 && v102 )
            j_j___libc_free_0_0(v102);
          if ( v66 > 0x40 && v64 )
            j_j___libc_free_0_0(v64);
          if ( v91 > 0x40 && v90 )
            j_j___libc_free_0_0(v90);
LABEL_48:
          if ( v89 > 0x40 && v88 )
            j_j___libc_free_0_0(v88);
          if ( (unsigned int)v99 > 0x40 && v98 )
            j_j___libc_free_0_0(v98);
          if ( (unsigned int)v97 > 0x40 && v96 )
            j_j___libc_free_0_0(v96);
          if ( !v79 )
            goto LABEL_35;
          goto LABEL_58;
        }
        v54 = v90;
      }
      else
      {
        v54 = v84;
      }
      v55 = ~v54;
      v56 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v53;
      v48 = v53 == 0;
      v57 = 0;
      if ( !v48 )
        v57 = v56;
      v90 = v57 & v55;
      goto LABEL_124;
    }
    v44 = v87;
    v91 = v87;
    if ( v87 > 0x40 )
    {
      sub_C43780((__int64)&v90, (const void **)&v86);
      v44 = v91;
      if ( v91 > 0x40 )
      {
        sub_C43D10((__int64)&v90);
LABEL_96:
        sub_C46250((__int64)&v90);
        v50 = v91;
        v51 = *a1;
        v91 = 0;
        v65 = v50;
        v93 = v50;
        v62 = v90;
        v92 = v90;
        v52 = sub_DBB9F0(v51, (__int64)v70, 0, 0);
        LODWORD(v103) = *(_DWORD *)(v52 + 8);
        if ( (unsigned int)v103 > 0x40 )
        {
          v72 = v52;
          sub_C43780((__int64)&v102, (const void **)v52);
          v52 = v72;
        }
        else
        {
          v102 = *(_QWORD *)v52;
        }
        LODWORD(v105) = *(_DWORD *)(v52 + 24);
        if ( (unsigned int)v105 > 0x40 )
          sub_C43780((__int64)&v104, (const void **)(v52 + 16));
        else
          v104 = *(_QWORD *)(v52 + 16);
        sub_AB0910((__int64)&v94, (__int64)&v102);
        v78 = (int)sub_C49970((__int64)&v92, &v94) >= 0;
        if ( v95 > 0x40 && v94 )
          j_j___libc_free_0_0(v94);
        if ( (unsigned int)v105 > 0x40 && v104 )
          j_j___libc_free_0_0(v104);
        if ( (unsigned int)v103 > 0x40 && v102 )
          j_j___libc_free_0_0(v102);
        if ( v65 > 0x40 && v62 )
          j_j___libc_free_0_0(v62);
        if ( v91 > 0x40 && v90 )
          j_j___libc_free_0_0(v90);
        goto LABEL_72;
      }
      v45 = v90;
    }
    else
    {
      v45 = v86;
    }
    v46 = ~v45;
    v47 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
    v48 = v44 == 0;
    v49 = 0;
    if ( !v48 )
      v49 = v47;
    v90 = v49 & v46;
    goto LABEL_96;
  }
  return result;
}
