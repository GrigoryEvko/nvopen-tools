// Function: sub_16AB900
// Address: 0x16ab900
//
unsigned __int64 *__fastcall sub_16AB900(unsigned __int64 *a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // ebx
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // r15d
  unsigned int v10; // ebx
  unsigned __int64 v11; // r15
  unsigned int v12; // eax
  unsigned int v13; // ebx
  unsigned __int64 v14; // r14
  unsigned int v15; // ebx
  _QWORD *v16; // r14
  unsigned int v17; // ebx
  unsigned __int64 v18; // r13
  unsigned int v19; // ecx
  unsigned int v20; // edx
  unsigned __int64 v21; // rax
  unsigned int v22; // ecx
  unsigned int v23; // eax
  unsigned int v24; // edx
  unsigned __int64 v25; // rax
  unsigned int v26; // ecx
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  unsigned int v29; // ecx
  unsigned int v30; // eax
  unsigned int v31; // edx
  unsigned __int64 v32; // rax
  unsigned int v33; // ebx
  const void *v34; // r14
  int v35; // eax
  bool v36; // al
  _QWORD *v37; // rax
  unsigned __int64 *v38; // rax
  unsigned int v39; // r13d
  unsigned __int64 v40; // r15
  bool v41; // cc
  unsigned int v42; // edx
  unsigned __int64 v43; // rsi
  unsigned int v45; // ebx
  unsigned __int64 v46; // r14
  unsigned int v47; // ebx
  unsigned __int64 v48; // r14
  unsigned int v49; // ebx
  const void *v50; // r14
  unsigned int v51; // ebx
  _QWORD *v52; // r14
  int v53; // eax
  unsigned int v54; // ecx
  unsigned __int64 v55; // rdx
  unsigned __int64 *v56; // rax
  unsigned int v57; // r12d
  unsigned __int64 v58; // r13
  unsigned int v60; // [rsp+24h] [rbp-11Ch]
  int v61; // [rsp+38h] [rbp-108h]
  int v62; // [rsp+38h] [rbp-108h]
  unsigned int v63; // [rsp+3Ch] [rbp-104h]
  unsigned int v64; // [rsp+3Ch] [rbp-104h]
  unsigned int v65; // [rsp+3Ch] [rbp-104h]
  unsigned __int64 v66; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v67; // [rsp+58h] [rbp-E8h]
  unsigned __int64 v68; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v69; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v70; // [rsp+70h] [rbp-D0h] BYREF
  int v71; // [rsp+78h] [rbp-C8h]
  const void *v72; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v73; // [rsp+88h] [rbp-B8h]
  _QWORD *v74; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v75; // [rsp+98h] [rbp-A8h]
  unsigned __int64 v76; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v77; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v78; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v79; // [rsp+B8h] [rbp-88h]
  unsigned __int64 v80; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v81; // [rsp+C8h] [rbp-78h]
  __int64 v82; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v83; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v84; // [rsp+E0h] [rbp-60h] BYREF
  unsigned int v85; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v86; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v87; // [rsp+F8h] [rbp-48h]
  unsigned __int64 v88; // [rsp+100h] [rbp-40h] BYREF
  unsigned int v89; // [rsp+108h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 8);
  v3 = v2 - 1;
  v83 = v2;
  v67 = 1;
  v66 = 0;
  v69 = 1;
  v4 = 1LL << ((unsigned __int8)v2 - 1);
  v68 = 0;
  v71 = 1;
  v70 = 0;
  v73 = 1;
  v72 = 0;
  v75 = 1;
  v74 = 0;
  v77 = 1;
  v76 = 0;
  v79 = 1;
  v78 = 0;
  v81 = 1;
  v80 = 0;
  if ( v2 <= 0x40 )
  {
    v82 = 0;
LABEL_3:
    v82 |= v4;
    goto LABEL_4;
  }
  sub_16A4EF0((__int64)&v82, 0, 0);
  if ( v83 <= 0x40 )
    goto LABEL_3;
  *(_QWORD *)(v82 + 8LL * (v3 >> 6)) |= v4;
LABEL_4:
  *((_DWORD *)a1 + 2) = 1;
  *a1 = 0;
  v5 = *(_DWORD *)(a2 + 8);
  v6 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 > 0x40 )
  {
    if ( (*(_QWORD *)(*(_QWORD *)a2 + 8LL * ((v5 - 1) >> 6)) & v6) == 0 )
    {
      v87 = *(_DWORD *)(a2 + 8);
      sub_16A4FD0((__int64)&v86, (const void **)a2);
      goto LABEL_9;
    }
    v89 = *(_DWORD *)(a2 + 8);
    sub_16A4FD0((__int64)&v88, (const void **)a2);
    LOBYTE(v5) = v89;
    if ( v89 > 0x40 )
    {
      sub_16A8F40((__int64 *)&v88);
      goto LABEL_8;
    }
    v7 = v88;
  }
  else
  {
    v7 = *(_QWORD *)a2;
    if ( (*(_QWORD *)a2 & v6) == 0 )
    {
      v87 = *(_DWORD *)(a2 + 8);
      v86 = v7;
      goto LABEL_9;
    }
    v89 = *(_DWORD *)(a2 + 8);
  }
  v88 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v7;
LABEL_8:
  sub_16A7400((__int64)&v88);
  v87 = v89;
  v86 = v88;
LABEL_9:
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  v66 = v86;
  v67 = v87;
  v8 = *(_DWORD *)(a2 + 8);
  v89 = v8;
  v9 = v8 - 1;
  if ( v8 > 0x40 )
  {
    sub_16A4FD0((__int64)&v88, (const void **)a2);
    v8 = v89;
    if ( v89 > 0x40 )
    {
      sub_16A8110((__int64)&v88, v9);
      goto LABEL_16;
    }
  }
  else
  {
    v88 = *(_QWORD *)a2;
  }
  if ( v9 == v8 )
    v88 = 0;
  else
    v88 >>= v9;
LABEL_16:
  sub_16A7200((__int64)&v88, &v82);
  v10 = v89;
  v89 = 0;
  v11 = v88;
  if ( v81 > 0x40 && v80 )
  {
    j_j___libc_free_0_0(v80);
    v80 = v11;
    v81 = v10;
    if ( v89 > 0x40 && v88 )
      j_j___libc_free_0_0(v88);
  }
  else
  {
    v80 = v88;
    v81 = v10;
  }
  sub_16AB0A0((__int64)&v88, (__int64)&v80, (__int64)&v66);
  v85 = v81;
  if ( v81 > 0x40 )
    sub_16A4FD0((__int64)&v84, (const void **)&v80);
  else
    v84 = v80;
  sub_16A7800((__int64)&v84, 1u);
  v12 = v85;
  v85 = 0;
  v87 = v12;
  v86 = v84;
  if ( v89 > 0x40 )
    sub_16A8F40((__int64 *)&v88);
  else
    v88 = ~v88 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v89);
  sub_16A7400((__int64)&v88);
  sub_16A7200((__int64)&v88, (__int64 *)&v86);
  v13 = v89;
  v89 = 0;
  v14 = v88;
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  v68 = v14;
  v69 = v13;
  if ( v87 > 0x40 && v86 )
    j_j___libc_free_0_0(v86);
  if ( v85 > 0x40 && v84 )
    j_j___libc_free_0_0(v84);
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  v61 = *(_DWORD *)(a2 + 8) - 1;
  sub_16A9D70((__int64)&v88, (__int64)&v82, (__int64)&v68);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  v72 = (const void *)v88;
  v73 = v89;
  sub_16A7B50((__int64)&v88, (__int64)&v72, (__int64 *)&v68);
  if ( v89 > 0x40 )
    sub_16A8F40((__int64 *)&v88);
  else
    v88 = ~v88 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v89);
  sub_16A7400((__int64)&v88);
  sub_16A7200((__int64)&v88, &v82);
  v15 = v89;
  v89 = 0;
  v16 = (_QWORD *)v88;
  if ( v75 > 0x40 && v74 )
  {
    j_j___libc_free_0_0(v74);
    v74 = v16;
    v75 = v15;
    if ( v89 > 0x40 && v88 )
      j_j___libc_free_0_0(v88);
  }
  else
  {
    v74 = (_QWORD *)v88;
    v75 = v15;
  }
  sub_16A9D70((__int64)&v88, (__int64)&v82, (__int64)&v66);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  v76 = v88;
  v77 = v89;
  sub_16A7B50((__int64)&v88, (__int64)&v76, (__int64 *)&v66);
  if ( v89 > 0x40 )
    sub_16A8F40((__int64 *)&v88);
  else
    v88 = ~v88 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v89);
  sub_16A7400((__int64)&v88);
  sub_16A7200((__int64)&v88, &v82);
  v17 = v89;
  v89 = 0;
  v18 = v88;
  if ( v79 > 0x40 && v78 )
  {
    j_j___libc_free_0_0(v78);
    v78 = v18;
    v79 = v17;
    if ( v89 > 0x40 && v88 )
      j_j___libc_free_0_0(v88);
  }
  else
  {
    v78 = v88;
    v79 = v17;
  }
  v19 = v73;
  v62 = v61 + 1;
  v63 = 1;
  v89 = v73;
  if ( v73 > 0x40 )
    goto LABEL_151;
LABEL_58:
  v20 = v19;
  v88 = (unsigned __int64)v72;
LABEL_59:
  v21 = 0;
  if ( v19 != 1 )
    v21 = (2 * v88) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v19);
  v88 = v21;
  while ( 1 )
  {
    if ( v20 > 0x40 && v72 )
      j_j___libc_free_0_0(v72);
    v22 = v75;
    v72 = (const void *)v88;
    v23 = v89;
    v89 = v75;
    v73 = v23;
    if ( v75 > 0x40 )
    {
      sub_16A4FD0((__int64)&v88, (const void **)&v74);
      v22 = v89;
      if ( v89 > 0x40 )
      {
        sub_16A7DC0((__int64 *)&v88, 1u);
        v24 = v75;
        goto LABEL_70;
      }
      v24 = v75;
    }
    else
    {
      v24 = v75;
      v88 = (unsigned __int64)v74;
    }
    v25 = 0;
    if ( v22 != 1 )
      v25 = (2 * v88) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
    v88 = v25;
LABEL_70:
    if ( v24 > 0x40 && v74 )
      j_j___libc_free_0_0(v74);
    v74 = (_QWORD *)v88;
    v75 = v89;
    if ( (int)sub_16A9900((__int64)&v74, &v68) < 0 )
      goto LABEL_74;
    v89 = v73;
    if ( v73 > 0x40 )
      sub_16A4FD0((__int64)&v88, &v72);
    else
      v88 = (unsigned __int64)v72;
    sub_16A7490((__int64)&v88, 1);
    v49 = v89;
    v89 = 0;
    v50 = (const void *)v88;
    if ( v73 > 0x40 && v72 )
    {
      j_j___libc_free_0_0(v72);
      v72 = v50;
      v73 = v49;
      if ( v89 > 0x40 && v88 )
        j_j___libc_free_0_0(v88);
      v89 = v75;
      if ( v75 > 0x40 )
      {
LABEL_192:
        sub_16A4FD0((__int64)&v88, (const void **)&v74);
        goto LABEL_177;
      }
    }
    else
    {
      v72 = (const void *)v88;
      v73 = v49;
      v89 = v75;
      if ( v75 > 0x40 )
        goto LABEL_192;
    }
    v88 = (unsigned __int64)v74;
LABEL_177:
    sub_16A7590((__int64)&v88, (__int64 *)&v68);
    v51 = v89;
    v89 = 0;
    v52 = (_QWORD *)v88;
    if ( v75 <= 0x40 || !v74 )
    {
      v74 = (_QWORD *)v88;
      v75 = v51;
LABEL_74:
      v26 = v77;
      v89 = v77;
      if ( v77 <= 0x40 )
        goto LABEL_75;
      goto LABEL_182;
    }
    j_j___libc_free_0_0(v74);
    v74 = v52;
    v75 = v51;
    if ( v89 <= 0x40 || !v88 )
      goto LABEL_74;
    j_j___libc_free_0_0(v88);
    v26 = v77;
    v89 = v77;
    if ( v77 <= 0x40 )
    {
LABEL_75:
      v27 = v26;
      v88 = v76;
      goto LABEL_76;
    }
LABEL_182:
    sub_16A4FD0((__int64)&v88, (const void **)&v76);
    v26 = v89;
    if ( v89 > 0x40 )
    {
      sub_16A7DC0((__int64 *)&v88, 1u);
      v27 = v77;
      goto LABEL_79;
    }
    v27 = v77;
LABEL_76:
    v28 = 0;
    if ( v26 != 1 )
      v28 = (2 * v88) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v26);
    v88 = v28;
LABEL_79:
    if ( v27 > 0x40 && v76 )
      j_j___libc_free_0_0(v76);
    v29 = v79;
    v76 = v88;
    v30 = v89;
    v89 = v79;
    v77 = v30;
    if ( v79 > 0x40 )
    {
      sub_16A4FD0((__int64)&v88, (const void **)&v78);
      v29 = v89;
      if ( v89 > 0x40 )
      {
        sub_16A7DC0((__int64 *)&v88, 1u);
        v31 = v79;
        goto LABEL_87;
      }
      v31 = v79;
    }
    else
    {
      v31 = v79;
      v88 = v78;
    }
    v32 = 0;
    if ( v29 != 1 )
      v32 = (2 * v88) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29);
    v88 = v32;
LABEL_87:
    if ( v31 > 0x40 && v78 )
      j_j___libc_free_0_0(v78);
    v78 = v88;
    v79 = v89;
    if ( (int)sub_16A9900((__int64)&v78, &v66) < 0 )
      goto LABEL_91;
    v89 = v77;
    if ( v77 > 0x40 )
      sub_16A4FD0((__int64)&v88, (const void **)&v76);
    else
      v88 = v76;
    sub_16A7490((__int64)&v88, 1);
    v45 = v89;
    v89 = 0;
    v46 = v88;
    if ( v77 > 0x40 && v76 )
    {
      j_j___libc_free_0_0(v76);
      v76 = v46;
      v77 = v45;
      if ( v89 > 0x40 && v88 )
        j_j___libc_free_0_0(v88);
      v89 = v79;
      if ( v79 > 0x40 )
      {
LABEL_190:
        sub_16A4FD0((__int64)&v88, (const void **)&v78);
        goto LABEL_162;
      }
    }
    else
    {
      v76 = v88;
      v77 = v45;
      v89 = v79;
      if ( v79 > 0x40 )
        goto LABEL_190;
    }
    v88 = v78;
LABEL_162:
    sub_16A7590((__int64)&v88, (__int64 *)&v66);
    v47 = v89;
    v89 = 0;
    v48 = v88;
    if ( v79 <= 0x40 || !v78 )
    {
      v78 = v88;
      v79 = v47;
LABEL_91:
      v89 = v67;
      if ( v67 > 0x40 )
        goto LABEL_167;
      goto LABEL_92;
    }
    j_j___libc_free_0_0(v78);
    v78 = v48;
    v79 = v47;
    if ( v89 <= 0x40 || !v88 )
      goto LABEL_91;
    j_j___libc_free_0_0(v88);
    v89 = v67;
    if ( v67 > 0x40 )
    {
LABEL_167:
      sub_16A4FD0((__int64)&v88, (const void **)&v66);
      goto LABEL_93;
    }
LABEL_92:
    v88 = v66;
LABEL_93:
    sub_16A7590((__int64)&v88, (__int64 *)&v78);
    v33 = v89;
    v89 = 0;
    v34 = (const void *)v88;
    if ( v63 > 0x40 && v70 )
    {
      j_j___libc_free_0_0(v70);
      v70 = (unsigned __int64)v34;
      v71 = v33;
      if ( v89 > 0x40 && v88 )
        j_j___libc_free_0_0(v88);
    }
    else
    {
      v70 = v88;
      v71 = v33;
    }
    v35 = sub_16A9900((__int64)&v72, &v70);
    v19 = v73;
    if ( v35 < 0 )
      goto LABEL_150;
    if ( v73 <= 0x40 )
    {
      if ( v34 != v72 )
      {
        v89 = v77;
        if ( v77 > 0x40 )
          goto LABEL_198;
LABEL_105:
        v88 = v76;
        goto LABEL_106;
      }
    }
    else
    {
      v64 = v73;
      v36 = sub_16A5220((__int64)&v72, (const void **)&v70);
      v19 = v64;
      if ( !v36 )
        break;
    }
    v65 = v75;
    if ( v75 <= 0x40 )
    {
      v37 = v74;
      goto LABEL_103;
    }
    v60 = v19;
    v53 = sub_16A57B0((__int64)&v74);
    v19 = v60;
    if ( v65 - v53 > 0x40 )
      break;
    v37 = (_QWORD *)*v74;
LABEL_103:
    if ( v37 )
      break;
LABEL_150:
    ++v62;
    v63 = v33;
    v89 = v19;
    if ( v19 <= 0x40 )
      goto LABEL_58;
LABEL_151:
    sub_16A4FD0((__int64)&v88, &v72);
    v19 = v89;
    if ( v89 <= 0x40 )
    {
      v20 = v73;
      goto LABEL_59;
    }
    sub_16A7DC0((__int64 *)&v88, 1u);
    v20 = v73;
  }
  v89 = v77;
  if ( v77 <= 0x40 )
    goto LABEL_105;
LABEL_198:
  sub_16A4FD0((__int64)&v88, (const void **)&v76);
LABEL_106:
  sub_16A7490((__int64)&v88, 1);
  v38 = a1;
  v39 = v89;
  v89 = 0;
  v40 = v88;
  if ( *((_DWORD *)a1 + 2) > 0x40u && (v38 = a1, *a1) )
  {
    j_j___libc_free_0_0(*a1);
    v41 = v89 <= 0x40;
    *a1 = v40;
    *((_DWORD *)a1 + 2) = v39;
    if ( !v41 && v88 )
      j_j___libc_free_0_0(v88);
  }
  else
  {
    *v38 = v88;
    *((_DWORD *)v38 + 2) = v39;
  }
  v42 = *(_DWORD *)(a2 + 8);
  v43 = *(_QWORD *)a2;
  if ( v42 > 0x40 )
    v43 = *(_QWORD *)(v43 + 8LL * ((v42 - 1) >> 6));
  if ( (v43 & (1LL << ((unsigned __int8)v42 - 1))) != 0 )
  {
    v54 = *((_DWORD *)a1 + 2);
    v89 = v54;
    if ( v54 > 0x40 )
    {
      sub_16A4FD0((__int64)&v88, (const void **)a1);
      LOBYTE(v54) = v89;
      if ( v89 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v88);
LABEL_213:
        sub_16A7400((__int64)&v88);
        v56 = a1;
        v57 = v89;
        v89 = 0;
        v58 = v88;
        if ( *((_DWORD *)a1 + 2) > 0x40u && (v56 = a1, *a1) )
        {
          j_j___libc_free_0_0(*a1);
          v41 = v89 <= 0x40;
          *a1 = v58;
          *((_DWORD *)a1 + 2) = v57;
          if ( !v41 && v88 )
            j_j___libc_free_0_0(v88);
        }
        else
        {
          *v56 = v88;
          *((_DWORD *)v56 + 2) = v57;
        }
        v42 = *(_DWORD *)(a2 + 8);
        goto LABEL_114;
      }
      v55 = v88;
    }
    else
    {
      v55 = *a1;
    }
    v88 = ~v55 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v54);
    goto LABEL_213;
  }
LABEL_114:
  v41 = v83 <= 0x40;
  *((_DWORD *)a1 + 4) = v62 - v42;
  if ( !v41 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v79 > 0x40 && v78 )
    j_j___libc_free_0_0(v78);
  if ( v77 > 0x40 && v76 )
    j_j___libc_free_0_0(v76);
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  if ( v33 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  return a1;
}
