// Function: sub_2DA2800
// Address: 0x2da2800
//
_QWORD *__fastcall sub_2DA2800(_QWORD *a1, __int64 a2, unsigned __int8 *a3, unsigned __int8 *a4)
{
  int v8; // eax
  int v9; // eax
  int v11; // eax
  __int64 **v12; // r14
  __int64 v13; // rdi
  __int64 *v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int8 **v18; // rax
  unsigned __int8 **v19; // rdx
  unsigned __int8 *v20; // rax
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // rsi
  unsigned __int8 *v23; // rax
  unsigned __int8 v24; // dl
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r15
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 *v46; // rax
  __int64 v47; // r9
  __int64 v48; // rcx
  char v49; // al
  unsigned __int8 *v50; // r9
  __int64 v51; // r8
  char v52; // al
  unsigned __int8 *v53; // r8
  __int64 v54; // r9
  unsigned __int8 *v55; // rax
  __int64 v56; // rt1
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r9
  __int64 v60; // r8
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  char v68; // al
  char v69; // al
  unsigned __int8 *v70; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v71; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v72; // [rsp+10h] [rbp-F0h]
  __int64 v73; // [rsp+10h] [rbp-F0h]
  __int64 *v74; // [rsp+20h] [rbp-E0h]
  __int64 v75; // [rsp+28h] [rbp-D8h]
  __int64 v76; // [rsp+28h] [rbp-D8h]
  __int64 v77; // [rsp+30h] [rbp-D0h]
  unsigned __int8 *v78; // [rsp+38h] [rbp-C8h]
  __int64 *v79; // [rsp+40h] [rbp-C0h]
  unsigned __int8 *v80; // [rsp+40h] [rbp-C0h]
  int v81; // [rsp+40h] [rbp-C0h]
  __int64 v82; // [rsp+40h] [rbp-C0h]
  __int64 v83; // [rsp+40h] [rbp-C0h]
  unsigned __int8 *v84; // [rsp+48h] [rbp-B8h]
  __int64 v85; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v86; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v87; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v88; // [rsp+58h] [rbp-A8h]
  unsigned __int8 *v89; // [rsp+60h] [rbp-A0h]
  int v90; // [rsp+6Ch] [rbp-94h]
  char v91; // [rsp+73h] [rbp-8Dh] BYREF
  int v92; // [rsp+74h] [rbp-8Ch] BYREF
  unsigned __int8 *v93; // [rsp+78h] [rbp-88h] BYREF
  __int64 *v94; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int8 **v95; // [rsp+88h] [rbp-78h]
  __int64 *v96; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int8 **v97; // [rsp+98h] [rbp-68h]
  __int64 *v98; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int8 **v99; // [rsp+A8h] [rbp-58h]
  __int64 v100; // [rsp+B0h] [rbp-50h] BYREF
  unsigned __int8 **v101; // [rsp+B8h] [rbp-48h]
  __int64 *v102; // [rsp+C0h] [rbp-40h] BYREF
  unsigned __int8 **v103; // [rsp+C8h] [rbp-38h]

  v8 = *a3;
  if ( (unsigned int)(v8 - 42) <= 1 )
  {
    v11 = *a4;
    if ( (unsigned int)(v11 - 42) <= 1 )
    {
      v90 = 0;
    }
    else
    {
      v90 = 3;
      if ( (unsigned int)(v11 - 44) > 1 )
        goto LABEL_5;
    }
  }
  else
  {
    if ( (unsigned int)(v8 - 44) > 1 )
    {
LABEL_5:
      *a1 = 0;
      a1[1] = 0;
      return a1;
    }
    v9 = *a4;
    if ( (unsigned int)(v9 - 42) <= 1 )
    {
      v90 = 1;
    }
    else
    {
      if ( (unsigned int)(v9 - 44) > 1 )
        goto LABEL_5;
      v90 = 2;
    }
  }
  if ( (unsigned __int8)sub_920620((__int64)a3)
    && ((sub_B45210((__int64)a3) & 0x20) == 0 || (sub_B45210((__int64)a4) & 0x20) == 0) )
  {
    goto LABEL_5;
  }
  v12 = (a3[7] & 0x40) != 0 ? (__int64 **)*((_QWORD *)a3 - 1) : (__int64 **)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  v13 = (__int64)v12[4];
  if ( *(_BYTE *)v13 <= 0x1Cu )
    goto LABEL_5;
  v14 = (a4[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)a4 - 1) : (__int64 *)&a4[-32 * (*((_DWORD *)a4 + 1) & 0x7FFFFFF)];
  v15 = v14[4];
  if ( *(_BYTE *)v15 <= 0x1Cu )
    goto LABEL_5;
  v16 = *(_QWORD *)(v13 + 16);
  if ( !v16 )
    goto LABEL_5;
  if ( *(_QWORD *)(v16 + 8) )
    goto LABEL_5;
  v17 = *(_QWORD *)(v15 + 16);
  if ( !v17 )
    goto LABEL_5;
  v88 = *(unsigned __int8 **)(v17 + 8);
  if ( v88 )
    goto LABEL_5;
  v79 = v14;
  v85 = v14[4];
  v18 = (unsigned __int8 **)sub_986520(v13);
  v89 = *v18;
  v84 = v18[4];
  v19 = (unsigned __int8 **)sub_986520(v85);
  v20 = *v19;
  if ( *v19 != v89 && v19[4] != v89 )
  {
    if ( v20 != v84 && v19[4] != v84 )
      goto LABEL_5;
    v21 = v84;
    v84 = v89;
    v89 = v21;
  }
  if ( v89 == v20 )
    v20 = v19[4];
  v78 = v20;
  v22 = v20;
  if ( (v90 & 1) != 0 )
  {
    v23 = v84;
    v88 = v89;
    v84 = v22;
    v78 = v23;
    v89 = 0;
  }
  v24 = *(_BYTE *)*v12;
  v74 = *v12;
  v77 = *v79;
  if ( v24 <= 0x1Cu || *(_BYTE *)v77 <= 0x1Cu )
    goto LABEL_5;
  v25 = v74[2];
  if ( !v25 )
    goto LABEL_34;
  if ( *(_QWORD *)(v25 + 8) )
    goto LABEL_34;
  v44 = *(_QWORD *)(v77 + 16);
  if ( !v44 || *(_QWORD *)(v44 + 8) || (unsigned __int8)(v24 - 46) > 1u || (unsigned __int8)(*(_BYTE *)v77 - 46) > 1u )
    goto LABEL_34;
  v45 = sub_986520((__int64)v74);
  v86 = *(unsigned __int8 **)v45;
  v72 = *(unsigned __int8 **)(v45 + 32);
  v46 = (__int64 *)sub_986520(v77);
  v47 = v46[4];
  v48 = *v46;
  v102 = 0;
  v80 = (unsigned __int8 *)v47;
  v75 = v48;
  v103 = &v93;
  v49 = sub_10A7530(&v102, 15, v86);
  v50 = v80;
  v51 = (__int64)v72;
  if ( v49 )
  {
    v81 = 1;
    v86 = v93;
  }
  else
  {
    v71 = v80;
    v102 = 0;
    v103 = &v93;
    v69 = sub_10A7530(&v102, 15, v72);
    v51 = (__int64)v72;
    v81 = 0;
    v50 = v71;
    if ( v69 )
    {
      v81 = 1;
      v51 = (__int64)v93;
    }
  }
  v70 = v50;
  v73 = v51;
  v52 = sub_2DA10D0(v75);
  v53 = (unsigned __int8 *)v73;
  v54 = (__int64)v70;
  if ( v52 )
  {
    v81 ^= 3u;
    v75 = (__int64)v93;
  }
  else
  {
    v102 = 0;
    v103 = &v93;
    v68 = sub_10A7530(&v102, 15, v70);
    v54 = (__int64)v70;
    v53 = (unsigned __int8 *)v73;
    if ( v68 )
    {
      v81 ^= 3u;
      v54 = (__int64)v93;
    }
  }
  if ( v86 != (unsigned __int8 *)v75 && v86 != (unsigned __int8 *)v54 )
  {
    if ( (unsigned __int8 *)v75 != v53 && (unsigned __int8 *)v54 != v53 )
      goto LABEL_34;
    v55 = v86;
    v86 = v53;
    v53 = v55;
  }
  if ( (unsigned __int8 *)v75 != v86 )
    v54 = v75;
  if ( (v81 & 0xFFFFFFFD) == 1 )
  {
    v88 = v86;
    v56 = v54;
    v54 = (__int64)v53;
    v53 = (unsigned __int8 *)v56;
  }
  else
  {
    v89 = v86;
  }
  if ( v89 && v88 )
  {
    v76 = v54;
    v87 = v53;
    sub_2DA3130(&v96, a2, v89, v88);
    if ( v96 )
    {
      sub_2DA3130(&v98, a2, v87, v76);
      if ( v98 )
      {
        v92 = 1;
        v102 = v74;
        v94 = (__int64 *)v77;
        sub_2D9F630(&v100, (__int64)&v91, &v92, (__int64 *)&v102, (__int64 *)&v94);
        v60 = v100;
        *(_DWORD *)(v100 + 36) = v81;
        v102 = v96;
        v103 = v97;
        if ( v97 )
        {
          v82 = v60;
          sub_2DA2510((__int64)v97);
          v60 = v82;
        }
        sub_2D9F520(v60, (__int64 *)&v102, v57, v58, v60, v59);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        v64 = v100;
        v102 = v98;
        v103 = v99;
        if ( v99 )
        {
          v83 = v100;
          sub_2DA2510((__int64)v99);
          v64 = v83;
        }
        sub_2D9F520(v64, (__int64 *)&v102, v61, v62, v64, v63);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        v102 = (__int64 *)v100;
        v103 = v101;
        if ( v101 )
          sub_2DA2510((__int64)v101);
        sub_2DA2080(&v94, a2, (unsigned __int64)&v102, v65, v66, v67);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        if ( v101 )
          sub_A191D0((volatile signed __int32 *)v101);
      }
      else
      {
        v94 = 0;
        v95 = 0;
      }
      if ( v99 )
        sub_A191D0((volatile signed __int32 *)v99);
    }
    else
    {
      v94 = 0;
      v95 = 0;
    }
    if ( v97 )
      sub_A191D0((volatile signed __int32 *)v97);
    goto LABEL_35;
  }
LABEL_34:
  v94 = 0;
  v95 = 0;
LABEL_35:
  if ( v94 )
  {
    sub_2DA3130(&v96, a2, v84, v78);
    if ( v96 )
    {
      sub_2DA3130(&v98, a2, v89, v88);
      if ( v98 )
      {
        v93 = a3;
        v102 = (__int64 *)a4;
        v92 = 1;
        sub_2D9F630(&v100, (__int64)&v91, &v92, (__int64 *)&v93, (__int64 *)&v102);
        v30 = v100;
        *(_DWORD *)(v100 + 36) = v90;
        v102 = v98;
        v103 = v99;
        if ( v99 )
          sub_2DA2510((__int64)v99);
        sub_2D9F520(v30, (__int64 *)&v102, v26, v27, v28, v29);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        v35 = v100;
        v102 = v96;
        v103 = v97;
        if ( v97 )
          sub_2DA2510((__int64)v97);
        sub_2D9F520(v35, (__int64 *)&v102, v31, v32, v33, v34);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        v40 = v100;
        v102 = v94;
        v103 = v95;
        if ( v95 )
          sub_2DA2510((__int64)v95);
        sub_2D9F520(v40, (__int64 *)&v102, v36, v37, v38, v39);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        v102 = (__int64 *)v100;
        v103 = v101;
        if ( v101 )
          sub_2DA2510((__int64)v101);
        sub_2DA2080(a1, a2, (unsigned __int64)&v102, v41, v42, v43);
        if ( v103 )
          sub_A191D0((volatile signed __int32 *)v103);
        if ( v101 )
          sub_A191D0((volatile signed __int32 *)v101);
      }
      else
      {
        *a1 = 0;
        a1[1] = 0;
      }
      if ( v99 )
        sub_A191D0((volatile signed __int32 *)v99);
    }
    else
    {
      *a1 = 0;
      a1[1] = 0;
    }
    if ( v97 )
      sub_A191D0((volatile signed __int32 *)v97);
  }
  else
  {
    *a1 = 0;
    a1[1] = 0;
  }
  if ( v95 )
    sub_A191D0((volatile signed __int32 *)v95);
  return a1;
}
