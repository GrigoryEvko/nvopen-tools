// Function: sub_3039000
// Address: 0x3039000
//
__int64 __fastcall sub_3039000(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int16 *v6; // rax
  __int16 v7; // r15
  unsigned int v8; // r14d
  char v9; // al
  __int64 v10; // r11
  __int64 v11; // r12
  __int64 v12; // rsi
  _QWORD *v13; // rdi
  signed __int64 v14; // rdx
  _QWORD *v15; // r8
  __int64 v16; // rcx
  _QWORD *v17; // rax
  _QWORD *v18; // rsi
  unsigned __int64 v19; // rdx
  unsigned __int64 v21; // rdx
  unsigned int v22; // eax
  unsigned __int64 v23; // rcx
  unsigned int v24; // eax
  unsigned __int64 v25; // rcx
  unsigned int v26; // eax
  unsigned __int64 v27; // rcx
  unsigned int v28; // edx
  __int64 v29; // rax
  unsigned int v30; // edx
  __int64 v31; // rax
  unsigned int v32; // r14d
  unsigned __int64 v33; // r13
  unsigned __int64 v34; // rdi
  __int128 v35; // rax
  int v36; // r9d
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r15
  unsigned int v42; // edx
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // r9d
  _QWORD *v50; // rax
  int v51; // edx
  __int64 v52; // rbx
  unsigned int v53; // edx
  unsigned __int64 v54; // rbx
  __int64 v55; // rax
  unsigned int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  int v61; // r9d
  __int64 v62; // rax
  int v63; // edx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  int v68; // r9d
  __int128 v69; // rax
  int v70; // r9d
  unsigned int v71; // eax
  unsigned __int64 v72; // rcx
  unsigned int v73; // r14d
  __int64 v74; // r13
  unsigned __int64 v75; // rdx
  __int64 v76; // rcx
  unsigned __int64 v77; // rdx
  __int64 v78; // rcx
  __int128 v79; // [rsp-10h] [rbp-1A0h]
  __int128 v80; // [rsp-10h] [rbp-1A0h]
  __int128 v81; // [rsp-10h] [rbp-1A0h]
  int v82; // [rsp+10h] [rbp-180h]
  __int64 v83; // [rsp+18h] [rbp-178h]
  __int64 v84; // [rsp+20h] [rbp-170h]
  __int64 v85; // [rsp+28h] [rbp-168h]
  __int64 v86; // [rsp+30h] [rbp-160h]
  __int64 v87; // [rsp+30h] [rbp-160h]
  __int64 v88; // [rsp+38h] [rbp-158h]
  __int64 v89; // [rsp+38h] [rbp-158h]
  unsigned __int64 v90; // [rsp+38h] [rbp-158h]
  __int64 v91; // [rsp+38h] [rbp-158h]
  __int64 v92; // [rsp+80h] [rbp-110h] BYREF
  int v93; // [rsp+88h] [rbp-108h]
  unsigned __int64 v94; // [rsp+90h] [rbp-100h] BYREF
  unsigned int v95; // [rsp+98h] [rbp-F8h]
  unsigned __int64 v96; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v97; // [rsp+A8h] [rbp-E8h]
  unsigned __int64 v98; // [rsp+B0h] [rbp-E0h] BYREF
  unsigned int v99; // [rsp+B8h] [rbp-D8h]
  unsigned __int64 v100; // [rsp+C0h] [rbp-D0h] BYREF
  unsigned int v101; // [rsp+C8h] [rbp-C8h]
  unsigned __int64 v102; // [rsp+D0h] [rbp-C0h] BYREF
  unsigned int v103; // [rsp+D8h] [rbp-B8h]
  unsigned __int64 v104; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v105; // [rsp+E8h] [rbp-A8h]
  unsigned __int64 v106; // [rsp+F0h] [rbp-A0h] BYREF
  unsigned int v107; // [rsp+F8h] [rbp-98h]
  unsigned __int64 v108; // [rsp+100h] [rbp-90h] BYREF
  unsigned int v109; // [rsp+108h] [rbp-88h]
  unsigned __int64 v110; // [rsp+110h] [rbp-80h] BYREF
  unsigned int v111; // [rsp+118h] [rbp-78h]
  unsigned __int64 v112; // [rsp+120h] [rbp-70h] BYREF
  unsigned __int64 v113; // [rsp+128h] [rbp-68h]
  __int64 v114; // [rsp+130h] [rbp-60h]
  unsigned __int64 v115; // [rsp+138h] [rbp-58h]
  __int64 v116; // [rsp+140h] [rbp-50h]
  __int64 v117; // [rsp+148h] [rbp-48h]
  __int64 v118; // [rsp+150h] [rbp-40h]
  __int64 v119; // [rsp+158h] [rbp-38h]

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *v6;
  v88 = *((_QWORD *)v6 + 1);
  v9 = sub_307AB50(*v6, v88);
  v10 = v88;
  if ( v9 != 1 && v7 != 37 )
    return a2;
  v12 = *(_QWORD *)(a2 + 80);
  v92 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v92, v12, 1);
    v10 = v88;
  }
  v13 = *(_QWORD **)(a2 + 40);
  v93 = *(_DWORD *)(a2 + 72);
  v14 = *(unsigned int *)(a2 + 64);
  v15 = &v13[5 * v14];
  if ( !(v14 >> 2) )
  {
    v17 = v13;
LABEL_86:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_18;
        goto LABEL_89;
      }
      v75 = *(unsigned int *)(*v17 + 24LL);
      if ( (unsigned int)v75 > 0x33 )
        goto LABEL_9;
      v76 = 0x8001800001800LL;
      if ( !_bittest64(&v76, v75) )
        goto LABEL_9;
      v17 += 5;
    }
    v77 = *(unsigned int *)(*v17 + 24LL);
    if ( (unsigned int)v77 > 0x33 )
      goto LABEL_9;
    v78 = 0x8001800001800LL;
    if ( !_bittest64(&v78, v77) )
      goto LABEL_9;
    v17 += 5;
LABEL_89:
    v39 = *(unsigned int *)(*v17 + 24LL);
    if ( (unsigned int)v39 > 0x33 )
      goto LABEL_9;
    v40 = 0x8001800001800LL;
    if ( !_bittest64(&v40, v39) )
      goto LABEL_9;
LABEL_18:
    v95 = 1;
    v94 = 0;
    if ( !(unsigned __int8)sub_307AB50(v8, v10) )
    {
      if ( v7 != 37 )
        BUG();
      sub_302FF80((__int64)&v110, a2, 3);
      v22 = v111;
      LODWORD(v113) = v111;
      if ( v111 > 0x40 )
      {
        sub_C43780((__int64)&v112, (const void **)&v110);
        v22 = v113;
        if ( (unsigned int)v113 > 0x40 )
        {
          sub_C47690((__int64 *)&v112, 0x18u);
          goto LABEL_26;
        }
      }
      else
      {
        v112 = v110;
      }
      v23 = 0;
      if ( v22 != 24 && v22 )
        v23 = (v112 << 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
      v112 = v23;
LABEL_26:
      sub_302FF80((__int64)&v104, a2, 2);
      v24 = v105;
      v107 = v105;
      if ( v105 > 0x40 )
      {
        sub_C43780((__int64)&v106, (const void **)&v104);
        v24 = v107;
        if ( v107 > 0x40 )
        {
          sub_C47690((__int64 *)&v106, 0x10u);
          goto LABEL_32;
        }
      }
      else
      {
        v106 = v104;
      }
      v25 = 0;
      if ( v24 != 16 && v24 )
        v25 = (v106 << 16) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
      v106 = v25;
LABEL_32:
      sub_302FF80((__int64)&v98, a2, 1);
      v26 = v99;
      v101 = v99;
      if ( v99 > 0x40 )
      {
        sub_C43780((__int64)&v100, (const void **)&v98);
        v26 = v101;
        if ( v101 > 0x40 )
        {
          sub_C47690((__int64 *)&v100, 8u);
          goto LABEL_38;
        }
      }
      else
      {
        v100 = v98;
      }
      v27 = 0;
      if ( v26 != 8 && v26 )
        v27 = (v100 << 8) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v26);
      v100 = v27;
LABEL_38:
      sub_302FF80((__int64)&v96, a2, 0);
      v28 = v101;
      if ( v101 > 0x40 )
      {
        sub_C43BD0(&v100, (__int64 *)&v96);
        v28 = v101;
        v29 = v100;
      }
      else
      {
        v29 = v96 | v100;
        v100 |= v96;
      }
      v103 = v28;
      v30 = v107;
      v102 = v29;
      v101 = 0;
      if ( v107 > 0x40 )
      {
        sub_C43BD0(&v106, (__int64 *)&v102);
        v30 = v107;
        v31 = v106;
      }
      else
      {
        v31 = v106 | v29;
        v106 = v31;
      }
      v32 = v113;
      v109 = v30;
      v108 = v31;
      v107 = 0;
      if ( (unsigned int)v113 > 0x40 )
      {
        sub_C43BD0(&v112, (__int64 *)&v108);
        v32 = v113;
      }
      else
      {
        v112 |= v31;
      }
      v33 = v112;
      LODWORD(v113) = 0;
      if ( v95 > 0x40 && v94 )
        j_j___libc_free_0_0(v94);
      v94 = v33;
      v95 = v32;
      if ( v109 > 0x40 && v108 )
        j_j___libc_free_0_0(v108);
      if ( v103 > 0x40 && v102 )
        j_j___libc_free_0_0(v102);
      if ( v97 > 0x40 && v96 )
        j_j___libc_free_0_0(v96);
      if ( v101 > 0x40 && v100 )
        j_j___libc_free_0_0(v100);
      if ( v99 > 0x40 && v98 )
        j_j___libc_free_0_0(v98);
      if ( v107 > 0x40 && v106 )
        j_j___libc_free_0_0(v106);
      if ( v105 <= 0x40 )
        goto LABEL_68;
      v34 = v104;
      if ( !v104 )
        goto LABEL_68;
      goto LABEL_67;
    }
    sub_302FF80((__int64)&v110, a2, 1);
    v71 = v111;
    LODWORD(v113) = v111;
    if ( v111 > 0x40 )
    {
      sub_C43780((__int64)&v112, (const void **)&v110);
      v71 = v113;
      if ( (unsigned int)v113 > 0x40 )
      {
        sub_C47690((__int64 *)&v112, 0x10u);
        goto LABEL_99;
      }
    }
    else
    {
      v112 = v110;
    }
    v72 = 0;
    if ( v71 != 16 && v71 )
      v72 = (v112 << 16) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v71);
    v112 = v72;
LABEL_99:
    sub_302FF80((__int64)&v108, a2, 0);
    v73 = v113;
    if ( (unsigned int)v113 > 0x40 )
    {
      sub_C43BD0(&v112, (__int64 *)&v108);
      v73 = v113;
      v74 = v112;
    }
    else
    {
      v74 = v108 | v112;
      v112 |= v108;
    }
    LODWORD(v113) = 0;
    if ( v95 > 0x40 && v94 )
      j_j___libc_free_0_0(v94);
    v94 = v74;
    v95 = v73;
    if ( v109 <= 0x40 )
      goto LABEL_68;
    v34 = v108;
    if ( !v108 )
      goto LABEL_68;
LABEL_67:
    j_j___libc_free_0_0(v34);
LABEL_68:
    if ( (unsigned int)v113 > 0x40 && v112 )
      j_j___libc_free_0_0(v112);
    if ( v111 > 0x40 && v110 )
      j_j___libc_free_0_0(v110);
    *(_QWORD *)&v35 = sub_34007B0(a4, (unsigned int)&v94, (unsigned int)&v92, 7, 0, 0, 0);
    v11 = sub_33FAF80(
            a4,
            234,
            (unsigned int)&v92,
            **(unsigned __int16 **)(a2 + 48),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
            v36,
            v35);
    if ( v95 > 0x40 && v94 )
      j_j___libc_free_0_0(v94);
    goto LABEL_12;
  }
  v16 = 0x8001800001800LL;
  v17 = v13;
  v18 = &v13[20 * (v14 >> 2)];
  while ( 1 )
  {
    v19 = *(unsigned int *)(*v17 + 24LL);
    if ( (unsigned int)v19 > 0x33 || !_bittest64(&v16, v19) )
      break;
    v21 = *(unsigned int *)(v17[5] + 24LL);
    if ( (unsigned int)v21 > 0x33 || !_bittest64(&v16, v21) )
    {
      if ( v15 != v17 + 5 )
        goto LABEL_10;
      goto LABEL_18;
    }
    v37 = *(unsigned int *)(v17[10] + 24LL);
    if ( (unsigned int)v37 > 0x33 || !_bittest64(&v16, v37) )
    {
      v17 += 10;
      break;
    }
    v38 = *(unsigned int *)(v17[15] + 24LL);
    if ( (unsigned int)v38 > 0x33 || !_bittest64(&v16, v38) )
    {
      v17 += 15;
      break;
    }
    v17 += 20;
    if ( v18 == v17 )
    {
      v14 = 0xCCCCCCCCCCCCCCCDLL * (v15 - v17);
      goto LABEL_86;
    }
  }
LABEL_9:
  if ( v15 == v17 )
    goto LABEL_18;
LABEL_10:
  if ( v7 == 37 )
  {
    v41 = v13[6];
    v82 = v10;
    v89 = v13[1];
    v85 = v13[5];
    v86 = sub_33FAFB0(a4, *v13, v89, &v92, 7, 0);
    v90 = v42 | v89 & 0xFFFFFFFF00000000LL;
    v43 = sub_33FAFB0(a4, v85, v41, &v92, 7, 0);
    v112 = v86;
    v113 = v90;
    v115 = v44 | v41 & 0xFFFFFFFF00000000LL;
    v114 = v43;
    v45 = sub_3400BD0(a4, 13120, (unsigned int)&v92, 7, 0, 0, 0);
    v117 = v46;
    v116 = v45;
    v47 = sub_3400BD0(a4, 0, (unsigned int)&v92, 7, 0, 0, 0);
    v119 = v48;
    *((_QWORD *)&v79 + 1) = 4;
    *(_QWORD *)&v79 = &v112;
    v118 = v47;
    v87 = sub_33FC220(a4, 537, (unsigned int)&v92, 37, 0, v49, v79);
    v50 = *(_QWORD **)(a2 + 40);
    LODWORD(v85) = v51;
    v52 = v50[11];
    v83 = v50[15];
    v91 = v50[16];
    v84 = sub_33FAFB0(a4, v50[10], v52, &v92, 7, 0);
    v54 = v53 | v52 & 0xFFFFFFFF00000000LL;
    v55 = sub_33FAFB0(a4, v83, v91, &v92, 7, 0);
    v112 = v84;
    v114 = v55;
    v113 = v54;
    v115 = v56 | v91 & 0xFFFFFFFF00000000LL;
    v57 = sub_3400BD0(a4, 13120, (unsigned int)&v92, 7, 0, 0, 0);
    v117 = v58;
    v116 = v57;
    v59 = sub_3400BD0(a4, 0, (unsigned int)&v92, 7, 0, 0, 0);
    v119 = v60;
    *((_QWORD *)&v80 + 1) = 4;
    *(_QWORD *)&v80 = &v112;
    v118 = v59;
    v62 = sub_33FC220(a4, 537, (unsigned int)&v92, 37, 0, v61, v80);
    LODWORD(v115) = v63;
    v114 = v62;
    v112 = v87;
    LODWORD(v113) = v85;
    v64 = sub_3400BD0(a4, 21520, (unsigned int)&v92, 7, 0, 0, 0);
    v117 = v65;
    v116 = v64;
    v66 = sub_3400BD0(a4, 0, (unsigned int)&v92, 7, 0, 0, 0);
    v119 = v67;
    *((_QWORD *)&v81 + 1) = 4;
    *(_QWORD *)&v81 = &v112;
    v118 = v66;
    *(_QWORD *)&v69 = sub_33FC220(a4, 537, (unsigned int)&v92, 37, 0, v68, v81);
    v11 = sub_33FAF80(a4, 234, (unsigned int)&v92, v8, v82, v70, v69);
  }
  else
  {
    v11 = a2;
  }
LABEL_12:
  if ( v92 )
    sub_B91220((__int64)&v92, v92);
  return v11;
}
