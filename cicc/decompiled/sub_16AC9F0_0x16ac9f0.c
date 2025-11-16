// Function: sub_16AC9F0
// Address: 0x16ac9f0
//
__int64 __fastcall sub_16AC9F0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 v7; // r8
  __int64 v8; // r14
  __int64 v9; // r14
  unsigned int v10; // eax
  unsigned int v11; // r13d
  unsigned __int64 v12; // r14
  unsigned int v13; // r13d
  _QWORD *v14; // r14
  unsigned int v15; // r13d
  const void *v16; // r14
  unsigned int v17; // r14d
  int v18; // r13d
  unsigned int v19; // eax
  unsigned int v20; // r13d
  const void *v21; // r14
  unsigned int v22; // eax
  unsigned int v23; // r13d
  _QWORD *v24; // r14
  unsigned __int64 v25; // rdi
  unsigned int v26; // r14d
  const void *v27; // r13
  unsigned int v28; // eax
  unsigned int v29; // edx
  unsigned int v30; // eax
  unsigned int v31; // r13d
  unsigned __int64 v32; // r14
  unsigned int v33; // eax
  unsigned int v34; // eax
  unsigned int v35; // r13d
  const void *v36; // r14
  unsigned __int64 v37; // rdi
  unsigned int v38; // eax
  unsigned int v39; // r13d
  const void *v40; // r14
  _QWORD *v41; // rax
  __int64 v42; // rax
  unsigned int v43; // ebx
  unsigned __int64 v44; // r12
  bool v45; // cc
  unsigned int v47; // r13d
  const void *v48; // r14
  unsigned int v49; // r13d
  _QWORD *v50; // r14
  unsigned int v51; // r13d
  unsigned __int64 v52; // r14
  unsigned int v53; // eax
  unsigned int v54; // r13d
  const void *v55; // r14
  unsigned __int64 v57; // [rsp+38h] [rbp-148h]
  unsigned int i; // [rsp+44h] [rbp-13Ch]
  unsigned __int64 v59; // [rsp+50h] [rbp-130h]
  int v60; // [rsp+50h] [rbp-130h]
  unsigned int v61; // [rsp+50h] [rbp-130h]
  unsigned int v62; // [rsp+58h] [rbp-128h]
  unsigned int v63; // [rsp+5Ch] [rbp-124h]
  __int64 v64; // [rsp+78h] [rbp-108h]
  unsigned __int64 v65; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v66; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v67; // [rsp+90h] [rbp-F0h] BYREF
  int v68; // [rsp+98h] [rbp-E8h]
  const void *v69; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v70; // [rsp+A8h] [rbp-D8h]
  _QWORD *v71; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v72; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v73; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned int v74; // [rsp+C8h] [rbp-B8h]
  const void *v75; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v76; // [rsp+D8h] [rbp-A8h]
  unsigned __int64 v77; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned int v78; // [rsp+E8h] [rbp-98h]
  __int64 v79; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v80; // [rsp+F8h] [rbp-88h]
  unsigned __int64 v81; // [rsp+100h] [rbp-80h] BYREF
  unsigned int v82; // [rsp+108h] [rbp-78h]
  const void *v83; // [rsp+110h] [rbp-70h] BYREF
  unsigned int v84; // [rsp+118h] [rbp-68h]
  unsigned __int64 v85; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v86; // [rsp+128h] [rbp-58h]
  unsigned __int64 v87; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v88; // [rsp+138h] [rbp-48h]
  unsigned __int64 v89; // [rsp+140h] [rbp-40h] BYREF
  unsigned int v90; // [rsp+148h] [rbp-38h]

  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v4 = *(_DWORD *)(a2 + 8);
  v66 = 1;
  v65 = 0;
  v68 = 1;
  v67 = 0;
  v70 = 1;
  v69 = 0;
  v72 = 1;
  v71 = 0;
  v74 = 1;
  v73 = 0;
  v76 = 1;
  v75 = 0;
  v90 = v4;
  if ( v4 <= 0x40 )
  {
    v78 = v4;
    v89 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
LABEL_3:
    v77 = v89;
    goto LABEL_4;
  }
  sub_16A4EF0((__int64)&v89, -1, 1);
  v4 = v90;
  v78 = v90;
  if ( v90 <= 0x40 )
    goto LABEL_3;
  sub_16A4FD0((__int64)&v77, (const void **)&v89);
  v4 = v78;
  if ( v78 > 0x40 )
  {
    sub_16A8110((__int64)&v77, a3);
    goto LABEL_6;
  }
LABEL_4:
  if ( a3 == v4 )
    v77 = 0;
  else
    v77 >>= a3;
LABEL_6:
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  v5 = *(_DWORD *)(a2 + 8);
  v6 = v5 - 1;
  v80 = v5;
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 <= 0x40 )
  {
    v8 = 1LL << ((unsigned __int8)v5 - 1);
    v79 = 0;
LABEL_11:
    v79 |= v7;
    goto LABEL_12;
  }
  v64 = 1LL << ((unsigned __int8)v5 - 1);
  sub_16A4EF0((__int64)&v79, 0, 0);
  v7 = v64;
  if ( v80 <= 0x40 )
  {
    v5 = *(_DWORD *)(a2 + 8);
    v6 = v5 - 1;
    v8 = 1LL << ((unsigned __int8)v5 - 1);
    goto LABEL_11;
  }
  *(_QWORD *)(v79 + 8LL * (v6 >> 6)) |= v64;
  v5 = *(_DWORD *)(a2 + 8);
  v6 = v5 - 1;
  v8 = 1LL << ((unsigned __int8)v5 - 1);
LABEL_12:
  v82 = v5;
  v9 = ~v8;
  if ( v5 > 0x40 )
  {
    sub_16A4EF0((__int64)&v81, -1, 1);
    if ( v82 > 0x40 )
    {
      *(_QWORD *)(v81 + 8LL * (v6 >> 6)) &= v9;
      v86 = v78;
      if ( v78 <= 0x40 )
        goto LABEL_15;
LABEL_242:
      sub_16A4FD0((__int64)&v85, (const void **)&v77);
      goto LABEL_16;
    }
  }
  else
  {
    v81 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
  }
  v81 &= v9;
  v86 = v78;
  if ( v78 > 0x40 )
    goto LABEL_242;
LABEL_15:
  v85 = v77;
LABEL_16:
  sub_16A7590((__int64)&v85, (__int64 *)a2);
  v10 = v86;
  v86 = 0;
  v88 = v10;
  v87 = v85;
  sub_16AB0A0((__int64)&v89, (__int64)&v87, a2);
  if ( v90 > 0x40 )
    sub_16A8F40((__int64 *)&v89);
  else
    v89 = ~v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
  sub_16A7400((__int64)&v89);
  sub_16A7200((__int64)&v89, (__int64 *)&v77);
  v11 = v90;
  v90 = 0;
  v12 = v89;
  if ( v66 > 0x40 && v65 )
  {
    j_j___libc_free_0_0(v65);
    v65 = v12;
    v66 = v11;
    if ( v90 > 0x40 && v89 )
      j_j___libc_free_0_0(v89);
  }
  else
  {
    v65 = v89;
    v66 = v11;
  }
  if ( v88 > 0x40 && v87 )
    j_j___libc_free_0_0(v87);
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  v63 = *(_DWORD *)(a2 + 8) - 1;
  sub_16A9D70((__int64)&v89, (__int64)&v79, (__int64)&v65);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  v69 = (const void *)v89;
  v70 = v90;
  sub_16A7B50((__int64)&v89, (__int64)&v69, (__int64 *)&v65);
  if ( v90 > 0x40 )
    sub_16A8F40((__int64 *)&v89);
  else
    v89 = ~v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
  sub_16A7400((__int64)&v89);
  sub_16A7200((__int64)&v89, &v79);
  v13 = v90;
  v90 = 0;
  v14 = (_QWORD *)v89;
  if ( v72 > 0x40 && v71 )
  {
    j_j___libc_free_0_0(v71);
    v71 = v14;
    v72 = v13;
    if ( v90 > 0x40 && v89 )
      j_j___libc_free_0_0(v89);
  }
  else
  {
    v71 = (_QWORD *)v89;
    v72 = v13;
  }
  sub_16A9D70((__int64)&v89, (__int64)&v81, a2);
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  v73 = v89;
  v74 = v90;
  sub_16A7B50((__int64)&v89, (__int64)&v73, (__int64 *)a2);
  if ( v90 > 0x40 )
    sub_16A8F40((__int64 *)&v89);
  else
    v89 = ~v89 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v90);
  sub_16A7400((__int64)&v89);
  sub_16A7200((__int64)&v89, (__int64 *)&v81);
  v15 = v90;
  v90 = 0;
  v16 = (const void *)v89;
  if ( v76 > 0x40 && v75 )
  {
    j_j___libc_free_0_0(v75);
    v75 = v16;
    v76 = v15;
    if ( v90 > 0x40 && v89 )
      j_j___libc_free_0_0(v89);
  }
  else
  {
    v75 = (const void *)v89;
    v76 = v15;
  }
  for ( i = 1; ; i = v39 )
  {
    ++v63;
    v88 = v66;
    if ( v66 > 0x40 )
      sub_16A4FD0((__int64)&v87, (const void **)&v65);
    else
      v87 = v65;
    sub_16A7590((__int64)&v87, (__int64 *)&v71);
    v17 = v88;
    v88 = 0;
    v90 = v17;
    v89 = v87;
    v59 = v87;
    v18 = sub_16A9900((__int64)&v71, &v89);
    if ( v17 > 0x40 )
    {
      if ( v59 )
      {
        j_j___libc_free_0_0(v59);
        if ( v88 > 0x40 )
        {
          if ( v87 )
            j_j___libc_free_0_0(v87);
        }
      }
    }
    if ( v18 >= 0 )
    {
      v88 = v70;
      if ( v70 > 0x40 )
        sub_16A4FD0((__int64)&v87, &v69);
      else
        v87 = (unsigned __int64)v69;
      sub_16A7200((__int64)&v87, (__int64 *)&v69);
      v19 = v88;
      v88 = 0;
      v90 = v19;
      v89 = v87;
      sub_16A7490((__int64)&v89, 1);
      v20 = v90;
      v90 = 0;
      v21 = (const void *)v89;
      if ( v70 > 0x40 && v69 )
      {
        j_j___libc_free_0_0(v69);
        v69 = v21;
        v70 = v20;
        if ( v90 > 0x40 && v89 )
          j_j___libc_free_0_0(v89);
      }
      else
      {
        v69 = (const void *)v89;
        v70 = v20;
      }
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      v88 = v72;
      if ( v72 > 0x40 )
        sub_16A4FD0((__int64)&v87, (const void **)&v71);
      else
        v87 = (unsigned __int64)v71;
      sub_16A7200((__int64)&v87, (__int64 *)&v71);
      v22 = v88;
      v88 = 0;
      v90 = v22;
      v89 = v87;
      sub_16A7590((__int64)&v89, (__int64 *)&v65);
      v23 = v90;
      v90 = 0;
      v24 = (_QWORD *)v89;
      if ( v72 > 0x40 && v71 )
      {
        j_j___libc_free_0_0(v71);
        v71 = v24;
        v72 = v23;
        if ( v90 > 0x40 && v89 )
          j_j___libc_free_0_0(v89);
      }
      else
      {
        v71 = (_QWORD *)v89;
        v72 = v23;
      }
      if ( v88 > 0x40 )
      {
        v25 = v87;
        if ( v87 )
LABEL_77:
          j_j___libc_free_0_0(v25);
      }
LABEL_78:
      v84 = v76;
      if ( v76 <= 0x40 )
        goto LABEL_79;
      goto LABEL_194;
    }
    v90 = v70;
    if ( v70 > 0x40 )
      sub_16A4FD0((__int64)&v89, &v69);
    else
      v89 = (unsigned __int64)v69;
    sub_16A7200((__int64)&v89, (__int64 *)&v69);
    v47 = v90;
    v90 = 0;
    v48 = (const void *)v89;
    if ( v70 > 0x40 && v69 )
    {
      j_j___libc_free_0_0(v69);
      v69 = v48;
      v70 = v47;
      if ( v90 > 0x40 && v89 )
        j_j___libc_free_0_0(v89);
      v90 = v72;
      if ( v72 > 0x40 )
      {
LABEL_221:
        sub_16A4FD0((__int64)&v89, (const void **)&v71);
        goto LABEL_189;
      }
    }
    else
    {
      v69 = (const void *)v89;
      v70 = v47;
      v90 = v72;
      if ( v72 > 0x40 )
        goto LABEL_221;
    }
    v89 = (unsigned __int64)v71;
LABEL_189:
    sub_16A7200((__int64)&v89, (__int64 *)&v71);
    v49 = v90;
    v90 = 0;
    v50 = (_QWORD *)v89;
    if ( v72 <= 0x40 || !v71 )
    {
      v71 = (_QWORD *)v89;
      v72 = v49;
      goto LABEL_78;
    }
    j_j___libc_free_0_0(v71);
    v71 = v50;
    v72 = v49;
    if ( v90 <= 0x40 )
      goto LABEL_78;
    v25 = v89;
    if ( v89 )
      goto LABEL_77;
    v84 = v76;
    if ( v76 <= 0x40 )
    {
LABEL_79:
      v83 = v75;
      goto LABEL_80;
    }
LABEL_194:
    sub_16A4FD0((__int64)&v83, &v75);
LABEL_80:
    sub_16A7490((__int64)&v83, 1);
    v26 = v84;
    v27 = v83;
    v84 = 0;
    v28 = *(_DWORD *)(a2 + 8);
    v86 = v26;
    v85 = (unsigned __int64)v83;
    v88 = v28;
    if ( v28 > 0x40 )
      sub_16A4FD0((__int64)&v87, (const void **)a2);
    else
      v87 = *(_QWORD *)a2;
    sub_16A7590((__int64)&v87, (__int64 *)&v75);
    v29 = v88;
    v88 = 0;
    v90 = v29;
    v62 = v29;
    v89 = v87;
    v57 = v87;
    v60 = sub_16A9900((__int64)&v85, &v89);
    if ( v62 > 0x40 )
    {
      if ( v57 )
      {
        j_j___libc_free_0_0(v57);
        if ( v88 > 0x40 )
        {
          if ( v87 )
            j_j___libc_free_0_0(v87);
        }
      }
    }
    if ( v26 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    if ( v60 >= 0 )
    {
      if ( (int)sub_16A9900((__int64)&v73, &v81) >= 0 )
        *(_BYTE *)(a1 + 16) = 1;
      v88 = v74;
      if ( v74 > 0x40 )
        sub_16A4FD0((__int64)&v87, (const void **)&v73);
      else
        v87 = v73;
      sub_16A7200((__int64)&v87, (__int64 *)&v73);
      v30 = v88;
      v88 = 0;
      v90 = v30;
      v89 = v87;
      sub_16A7490((__int64)&v89, 1);
      v31 = v90;
      v90 = 0;
      v32 = v89;
      if ( v74 > 0x40 && v73 )
      {
        j_j___libc_free_0_0(v73);
        v73 = v32;
        v74 = v31;
        if ( v90 > 0x40 && v89 )
          j_j___libc_free_0_0(v89);
      }
      else
      {
        v73 = v89;
        v74 = v31;
      }
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      v86 = v76;
      if ( v76 > 0x40 )
        sub_16A4FD0((__int64)&v85, &v75);
      else
        v85 = (unsigned __int64)v75;
      sub_16A7200((__int64)&v85, (__int64 *)&v75);
      v33 = v86;
      v86 = 0;
      v88 = v33;
      v87 = v85;
      sub_16A7490((__int64)&v87, 1);
      v34 = v88;
      v88 = 0;
      v90 = v34;
      v89 = v87;
      sub_16A7590((__int64)&v89, (__int64 *)a2);
      v35 = v90;
      v90 = 0;
      v36 = (const void *)v89;
      if ( v76 > 0x40 && v75 )
      {
        j_j___libc_free_0_0(v75);
        v75 = v36;
        v76 = v35;
        if ( v90 > 0x40 && v89 )
          j_j___libc_free_0_0(v89);
      }
      else
      {
        v75 = (const void *)v89;
        v76 = v35;
      }
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      if ( v86 > 0x40 )
      {
        v37 = v85;
        if ( v85 )
LABEL_118:
          j_j___libc_free_0_0(v37);
      }
LABEL_119:
      v88 = *(_DWORD *)(a2 + 8);
      if ( v88 > 0x40 )
        goto LABEL_214;
      goto LABEL_120;
    }
    if ( (int)sub_16A9900((__int64)&v73, (unsigned __int64 *)&v79) >= 0 )
      *(_BYTE *)(a1 + 16) = 1;
    v90 = v74;
    if ( v74 > 0x40 )
      sub_16A4FD0((__int64)&v89, (const void **)&v73);
    else
      v89 = v73;
    sub_16A7200((__int64)&v89, (__int64 *)&v73);
    v51 = v90;
    v90 = 0;
    v52 = v89;
    if ( v74 > 0x40 && v73 )
    {
      j_j___libc_free_0_0(v73);
      v73 = v52;
      v74 = v51;
      if ( v90 > 0x40 && v89 )
        j_j___libc_free_0_0(v89);
      v88 = v76;
      if ( v76 > 0x40 )
      {
LABEL_219:
        sub_16A4FD0((__int64)&v87, &v75);
        goto LABEL_206;
      }
    }
    else
    {
      v73 = v89;
      v74 = v51;
      v88 = v76;
      if ( v76 > 0x40 )
        goto LABEL_219;
    }
    v87 = (unsigned __int64)v75;
LABEL_206:
    sub_16A7200((__int64)&v87, (__int64 *)&v75);
    v53 = v88;
    v88 = 0;
    v90 = v53;
    v89 = v87;
    sub_16A7490((__int64)&v89, 1);
    v54 = v90;
    v90 = 0;
    v55 = (const void *)v89;
    if ( v76 > 0x40 && v75 )
    {
      j_j___libc_free_0_0(v75);
      v75 = v55;
      v76 = v54;
      if ( v90 > 0x40 && v89 )
        j_j___libc_free_0_0(v89);
    }
    else
    {
      v75 = (const void *)v89;
      v76 = v54;
    }
    if ( v88 <= 0x40 )
      goto LABEL_119;
    v37 = v87;
    if ( v87 )
      goto LABEL_118;
    v88 = *(_DWORD *)(a2 + 8);
    if ( v88 > 0x40 )
    {
LABEL_214:
      sub_16A4FD0((__int64)&v87, (const void **)a2);
      goto LABEL_121;
    }
LABEL_120:
    v87 = *(_QWORD *)a2;
LABEL_121:
    sub_16A7800((__int64)&v87, 1u);
    v38 = v88;
    v88 = 0;
    v90 = v38;
    v89 = v87;
    sub_16A7590((__int64)&v89, (__int64 *)&v75);
    v39 = v90;
    v90 = 0;
    v40 = (const void *)v89;
    if ( i <= 0x40 )
      goto LABEL_174;
    if ( v67 )
    {
      j_j___libc_free_0_0(v67);
      v67 = (unsigned __int64)v40;
      v68 = v39;
      if ( v90 > 0x40 && v89 )
        j_j___libc_free_0_0(v89);
    }
    else
    {
LABEL_174:
      v67 = v89;
      v68 = v39;
    }
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0(v87);
    if ( 2 * *(_DWORD *)(a2 + 8) <= v63 )
      break;
    if ( (int)sub_16A9900((__int64)&v69, &v67) >= 0 )
    {
      if ( v70 <= 0x40 )
      {
        if ( v40 != v69 )
          break;
      }
      else if ( !sub_16A5220((__int64)&v69, (const void **)&v67) )
      {
        break;
      }
      v61 = v72;
      if ( v72 > 0x40 )
      {
        if ( v61 - (unsigned int)sub_16A57B0((__int64)&v71) > 0x40 )
          break;
        v41 = (_QWORD *)*v71;
      }
      else
      {
        v41 = v71;
      }
      if ( v41 )
        break;
    }
  }
  v90 = v74;
  if ( v74 > 0x40 )
    sub_16A4FD0((__int64)&v89, (const void **)&v73);
  else
    v89 = v73;
  sub_16A7490((__int64)&v89, 1);
  v42 = a1;
  v43 = v90;
  v90 = 0;
  v44 = v89;
  if ( *(_DWORD *)(a1 + 8) > 0x40u && (v42 = a1, *(_QWORD *)a1) )
  {
    j_j___libc_free_0_0(*(_QWORD *)a1);
    v45 = v90 <= 0x40;
    *(_QWORD *)a1 = v44;
    *(_DWORD *)(a1 + 8) = v43;
    if ( !v45 && v89 )
      j_j___libc_free_0_0(v89);
  }
  else
  {
    *(_QWORD *)v42 = v89;
    *(_DWORD *)(v42 + 8) = v43;
  }
  v45 = v82 <= 0x40;
  *(_DWORD *)(a1 + 20) = v63 - *(_DWORD *)(a2 + 8);
  if ( !v45 && v81 )
    j_j___libc_free_0_0(v81);
  if ( v80 > 0x40 && v79 )
    j_j___libc_free_0_0(v79);
  if ( v78 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  if ( v76 > 0x40 && v75 )
    j_j___libc_free_0_0(v75);
  if ( v74 > 0x40 && v73 )
    j_j___libc_free_0_0(v73);
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  if ( v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  if ( v39 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  return a1;
}
