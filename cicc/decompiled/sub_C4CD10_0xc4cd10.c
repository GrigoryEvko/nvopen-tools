// Function: sub_C4CD10
// Address: 0xc4cd10
//
__int64 __fastcall sub_C4CD10(__int64 a1, const void **a2, __int64 *a3, unsigned __int64 *a4, unsigned int a5)
{
  unsigned int v7; // ebx
  __int64 v8; // rax
  unsigned int v10; // ebx
  unsigned int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // ebx
  __int64 v18; // rdx
  unsigned int v19; // ecx
  __int64 v20; // rax
  bool v21; // al
  unsigned int v22; // eax
  __int64 v23; // r8
  unsigned int v24; // eax
  unsigned int v25; // eax
  __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  unsigned int v28; // eax
  unsigned int v29; // eax
  __int64 v30; // rdi
  unsigned int v31; // edx
  int v32; // eax
  bool v33; // al
  unsigned int v34; // eax
  __int64 v35; // rax
  unsigned int v36; // eax
  unsigned int v37; // eax
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rdx
  unsigned int v40; // eax
  unsigned __int64 v41; // rax
  unsigned int v42; // eax
  unsigned int v43; // edx
  unsigned __int64 v44; // rax
  unsigned int v45; // edx
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rcx
  unsigned int v48; // eax
  unsigned int v49; // eax
  unsigned int v50; // eax
  unsigned int v51; // ebx
  __int64 v52; // r12
  unsigned int v53; // r13d
  __int64 v54; // rcx
  __int64 v55; // rdi
  char v56; // r13
  char v57; // al
  unsigned int v58; // eax
  __int64 v59; // rax
  unsigned int v60; // eax
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rdx
  unsigned int v63; // eax
  unsigned int v64; // eax
  bool v65; // [rsp+10h] [rbp-130h]
  unsigned int v66; // [rsp+10h] [rbp-130h]
  bool v67; // [rsp+20h] [rbp-120h]
  unsigned int v69; // [rsp+30h] [rbp-110h]
  __int64 v71; // [rsp+40h] [rbp-100h] BYREF
  unsigned int v72; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v73; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v74; // [rsp+58h] [rbp-E8h]
  __int64 v75; // [rsp+60h] [rbp-E0h] BYREF
  unsigned int v76; // [rsp+68h] [rbp-D8h]
  const void *v77; // [rsp+70h] [rbp-D0h] BYREF
  unsigned int v78; // [rsp+78h] [rbp-C8h]
  unsigned __int64 v79; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v80; // [rsp+88h] [rbp-B8h]
  const void *v81; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v82; // [rsp+98h] [rbp-A8h]
  __int64 v83; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v84; // [rsp+A8h] [rbp-98h]
  __int64 v85; // [rsp+B0h] [rbp-90h] BYREF
  unsigned int v86; // [rsp+B8h] [rbp-88h]
  unsigned __int64 v87; // [rsp+C0h] [rbp-80h] BYREF
  unsigned int v88; // [rsp+C8h] [rbp-78h]
  char *v89; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v90; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v91; // [rsp+E0h] [rbp-60h] BYREF
  unsigned int v92; // [rsp+E8h] [rbp-58h]
  unsigned __int64 v93; // [rsp+F0h] [rbp-50h] BYREF
  unsigned int v94; // [rsp+F8h] [rbp-48h]
  unsigned __int64 v95; // [rsp+100h] [rbp-40h] BYREF
  unsigned int v96; // [rsp+108h] [rbp-38h]

  v7 = *((_DWORD *)a2 + 2);
  sub_C44B10((__int64)&v95, (char **)a4, a5);
  if ( v96 <= 0x40 )
  {
    v67 = v95 == 0;
  }
  else
  {
    v69 = v96;
    v67 = v69 == (unsigned int)sub_C444A0((__int64)&v95);
    if ( v95 )
      j_j___libc_free_0_0(v95);
  }
  if ( v67 )
  {
    v96 = v7;
    v8 = 0;
    if ( v7 > 0x40 )
    {
      sub_C43690((__int64)&v95, 0, 0);
      v8 = v95;
      v7 = v96;
    }
    *(_DWORD *)(a1 + 8) = v7;
    *(_QWORD *)a1 = v8;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  v10 = 3 * v7;
  sub_C44830((__int64)&v95, a2, v10);
  if ( *((_DWORD *)a2 + 2) > 0x40u && *a2 )
    j_j___libc_free_0_0(*a2);
  *a2 = (const void *)v95;
  *((_DWORD *)a2 + 2) = v96;
  sub_C44830((__int64)&v95, a3, v10);
  if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    j_j___libc_free_0_0(*a3);
  *a3 = v95;
  *((_DWORD *)a3 + 2) = v96;
  sub_C44830((__int64)&v95, a4, v10);
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
    j_j___libc_free_0_0(*a4);
  *a4 = v95;
  *((_DWORD *)a4 + 2) = v96;
  v11 = *((_DWORD *)a2 + 2);
  v12 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 > 0x40 )
  {
    if ( (*((_QWORD *)*a2 + ((v11 - 1) >> 6)) & v12) == 0 )
      goto LABEL_20;
    sub_C43D10((__int64)a2);
  }
  else
  {
    if ( ((unsigned __int64)*a2 & v12) == 0 )
      goto LABEL_20;
    v47 = ~(unsigned __int64)*a2 & (0xFFFFFFFFFFFFFFFFLL >> (63 - ((v11 - 1) & 0x3F)));
    if ( !v11 )
      v47 = 0;
    *a2 = (const void *)v47;
  }
  sub_C46250((__int64)a2);
  v43 = *((_DWORD *)a3 + 2);
  if ( v43 > 0x40 )
  {
    sub_C43D10((__int64)a3);
  }
  else
  {
    v44 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v43) & ~*a3;
    if ( !v43 )
      v44 = 0;
    *a3 = v44;
  }
  sub_C46250((__int64)a3);
  v45 = *((_DWORD *)a4 + 2);
  if ( v45 > 0x40 )
  {
    sub_C43D10((__int64)a4);
  }
  else
  {
    v46 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v45) & ~*a4;
    if ( !v45 )
      v46 = 0;
    *a4 = v46;
  }
  sub_C46250((__int64)a4);
LABEL_20:
  v72 = v10;
  v13 = 1LL << a5;
  if ( v10 > 0x40 )
  {
    sub_C43690((__int64)&v71, 0, 0);
    v13 = 1LL << a5;
    if ( v72 > 0x40 )
    {
      *(_QWORD *)(v71 + 8LL * (a5 >> 6)) |= 1LL << a5;
      v96 = *((_DWORD *)a2 + 2);
      if ( v96 <= 0x40 )
        goto LABEL_23;
      goto LABEL_157;
    }
  }
  else
  {
    v71 = 0;
  }
  v71 |= v13;
  v96 = *((_DWORD *)a2 + 2);
  if ( v96 <= 0x40 )
  {
LABEL_23:
    v95 = (unsigned __int64)*a2;
    goto LABEL_24;
  }
LABEL_157:
  sub_C43780((__int64)&v95, a2);
LABEL_24:
  sub_C47170((__int64)&v95, 2u);
  v74 = v96;
  v73 = v95;
  sub_C472A0((__int64)&v75, (__int64)a3, a3);
  v14 = *((_DWORD *)a3 + 2);
  v15 = 1LL << ((unsigned __int8)v14 - 1);
  if ( v14 > 0x40 )
    v16 = *(_QWORD *)(*a3 + 8LL * ((v14 - 1) >> 6));
  else
    v16 = *a3;
  if ( (v16 & v15) != 0 )
  {
    v92 = v74;
    if ( v74 > 0x40 )
      sub_C43780((__int64)&v91, (const void **)&v73);
    else
      v91 = v73;
    sub_C47170((__int64)&v91, 2u);
    v36 = v92;
    v92 = 0;
    v94 = v36;
    v93 = v91;
    sub_C4A1D0((__int64)&v95, (__int64)&v75, (__int64)&v93);
    if ( v96 > 0x40 )
    {
      sub_C43D10((__int64)&v95);
    }
    else
    {
      v95 = ~v95;
      sub_C43640(&v95);
    }
    sub_C46250((__int64)&v95);
    sub_C45EE0((__int64)&v95, (__int64 *)a4);
    v88 = v96;
    v87 = v95;
    if ( v94 > 0x40 && v93 )
      j_j___libc_free_0_0(v93);
    if ( v92 > 0x40 && v91 )
      j_j___libc_free_0_0(v91);
    sub_C4BD10((__int64)&v95, (__int64)&v87, (__int64)&v71);
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0(v87);
    v87 = v95;
    v88 = v96;
    if ( (int)sub_C4C880((__int64)a4, (__int64)&v87) <= 0 )
    {
      sub_C46B40((__int64)a4, (__int64 *)&v87);
LABEL_140:
      if ( v88 > 0x40 && v87 )
        j_j___libc_free_0_0(v87);
      goto LABEL_35;
    }
    v37 = *((_DWORD *)a4 + 2);
    v90 = v37;
    if ( v37 > 0x40 )
    {
      sub_C43780((__int64)&v89, (const void **)a4);
      v37 = v90;
      if ( v90 > 0x40 )
      {
        sub_C43D10((__int64)&v89);
LABEL_123:
        sub_C46250((__int64)&v89);
        v40 = v90;
        v90 = 0;
        v92 = v40;
        v91 = (unsigned __int64)v89;
        sub_C4BD10((__int64)&v93, (__int64)&v91, (__int64)&v71);
        if ( v94 > 0x40 )
        {
          sub_C43D10((__int64)&v93);
        }
        else
        {
          v41 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v94) & ~v93;
          if ( !v94 )
            v41 = 0;
          v93 = v41;
        }
        sub_C46250((__int64)&v93);
        v42 = v94;
        v94 = 0;
        v96 = v42;
        v95 = v93;
        sub_C46B40((__int64)a4, (__int64 *)&v95);
        if ( v96 > 0x40 && v95 )
          j_j___libc_free_0_0(v95);
        if ( v94 > 0x40 && v93 )
          j_j___libc_free_0_0(v93);
        if ( v92 > 0x40 && v91 )
          j_j___libc_free_0_0(v91);
        if ( v90 > 0x40 && v89 )
          j_j___libc_free_0_0(v89);
        v67 = 1;
        goto LABEL_140;
      }
      v38 = (unsigned __int64)v89;
    }
    else
    {
      v38 = *a4;
    }
    v39 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v37) & ~v38;
    if ( !v37 )
      v39 = 0;
    v89 = (char *)v39;
    goto LABEL_123;
  }
  sub_C4B8A0((__int64)&v95, (__int64)a4, (__int64)&v71);
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
    j_j___libc_free_0_0(*a4);
  v17 = v96;
  v18 = v95;
  v19 = v96 - 1;
  *a4 = v95;
  *((_DWORD *)a4 + 2) = v17;
  v20 = 1LL << v19;
  if ( v17 <= 0x40 )
  {
    if ( (v20 & v18) != 0 )
      goto LABEL_35;
    v21 = v18 == 0;
  }
  else
  {
    if ( (*(_QWORD *)(v18 + 8LL * (v19 >> 6)) & v20) != 0 )
      goto LABEL_35;
    v21 = v17 == (unsigned int)sub_C444A0((__int64)a4);
  }
  if ( !v21 )
  {
    sub_C46B40((__int64)a4, &v71);
    v67 = 0;
  }
LABEL_35:
  v92 = *((_DWORD *)a2 + 2);
  if ( v92 > 0x40 )
    sub_C43780((__int64)&v91, a2);
  else
    v91 = (unsigned __int64)*a2;
  sub_C47170((__int64)&v91, 4u);
  v22 = v92;
  v92 = 0;
  v94 = v22;
  v93 = v91;
  sub_C472A0((__int64)&v95, (__int64)&v93, (__int64 *)a4);
  if ( v96 > 0x40 )
  {
    sub_C43D10((__int64)&v95);
  }
  else
  {
    v95 = ~v95;
    sub_C43640(&v95);
  }
  sub_C46250((__int64)&v95);
  sub_C45EE0((__int64)&v95, &v75);
  v78 = v96;
  v77 = (const void *)v95;
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0(v91);
  sub_C4AAE0((__int64)&v79, &v77);
  sub_C472A0((__int64)&v81, (__int64)&v79, (__int64 *)&v79);
  if ( v82 <= 0x40 )
    v65 = v81 == v77;
  else
    v65 = sub_C43C50((__int64)&v81, &v77);
  if ( (int)sub_C4C880((__int64)&v81, (__int64)&v77) > 0 )
    sub_C46F20((__int64)&v79, 1u);
  v84 = 1;
  v83 = 0;
  v86 = 1;
  v85 = 0;
  if ( !v67 )
  {
    v60 = *((_DWORD *)a3 + 2);
    v92 = v60;
    if ( v60 > 0x40 )
    {
      sub_C43780((__int64)&v91, (const void **)a3);
      v60 = v92;
      if ( v92 > 0x40 )
      {
        sub_C43D10((__int64)&v91);
        goto LABEL_199;
      }
      v61 = v91;
    }
    else
    {
      v61 = *a3;
    }
    v62 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v60) & ~v61;
    if ( !v60 )
      v62 = 0;
    v91 = v62;
LABEL_199:
    sub_C46250((__int64)&v91);
    v63 = v92;
    v92 = 0;
    v94 = v63;
    v93 = v91;
    sub_C45EE0((__int64)&v93, (__int64 *)&v79);
    v64 = v94;
    v94 = 0;
    v96 = v64;
    v95 = v93;
    sub_C4C400((__int64)&v95, (__int64)&v73, (__int64)&v83, (__int64)&v85);
    if ( v96 <= 0x40 )
      goto LABEL_68;
    v30 = v95;
    if ( !v95 )
      goto LABEL_68;
    goto LABEL_67;
  }
  v92 = v80;
  v23 = !v65;
  if ( v80 > 0x40 )
  {
    sub_C43780((__int64)&v91, (const void **)&v79);
    v23 = !v65;
  }
  else
  {
    v91 = v79;
  }
  sub_C46A40((__int64)&v91, v23);
  v24 = v92;
  v92 = 0;
  v94 = v24;
  v93 = v91;
  v25 = *((_DWORD *)a3 + 2);
  v88 = v25;
  if ( v25 > 0x40 )
  {
    sub_C43780((__int64)&v87, (const void **)a3);
    v25 = v88;
    if ( v88 > 0x40 )
    {
      sub_C43D10((__int64)&v87);
      goto LABEL_57;
    }
    v26 = v87;
  }
  else
  {
    v26 = *a3;
  }
  v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v25) & ~v26;
  if ( !v25 )
    v27 = 0;
  v87 = v27;
LABEL_57:
  sub_C46250((__int64)&v87);
  v28 = v88;
  v88 = 0;
  v90 = v28;
  v89 = (char *)v87;
  if ( v94 > 0x40 )
  {
    sub_C43D10((__int64)&v93);
  }
  else
  {
    v93 = ~v93;
    sub_C43640(&v93);
  }
  sub_C46250((__int64)&v93);
  sub_C45EE0((__int64)&v93, (__int64 *)&v89);
  v29 = v94;
  v94 = 0;
  v96 = v29;
  v95 = v93;
  sub_C4C400((__int64)&v95, (__int64)&v73, (__int64)&v83, (__int64)&v85);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v90 > 0x40 && v89 )
    j_j___libc_free_0_0(v89);
  if ( v88 <= 0x40 )
    goto LABEL_68;
  v30 = v87;
  if ( !v87 )
    goto LABEL_68;
LABEL_67:
  j_j___libc_free_0_0(v30);
LABEL_68:
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0(v93);
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0(v91);
  if ( v65
    && ((v31 = v86, v86 <= 0x40)
      ? (v33 = v85 == 0)
      : (v66 = v86, v32 = sub_C444A0((__int64)&v85), v31 = v66, v33 = v66 == v32),
        v33) )
  {
    v34 = v84;
    v84 = 0;
    *(_DWORD *)(a1 + 8) = v34;
    v35 = v83;
    *(_BYTE *)(a1 + 16) = 1;
    *(_QWORD *)a1 = v35;
  }
  else
  {
    sub_C472A0((__int64)&v91, (__int64)a2, &v83);
    sub_C45EE0((__int64)&v91, a3);
    v48 = v92;
    v92 = 0;
    v94 = v48;
    v93 = v91;
    sub_C472A0((__int64)&v95, (__int64)&v93, &v83);
    sub_C45EE0((__int64)&v95, (__int64 *)a4);
    v88 = v96;
    v87 = v95;
    if ( v94 > 0x40 && v93 )
      j_j___libc_free_0_0(v93);
    if ( v92 > 0x40 && v91 )
      j_j___libc_free_0_0(v91);
    sub_C472A0((__int64)&v91, (__int64)&v73, &v83);
    sub_C45EE0((__int64)&v91, (__int64 *)&v87);
    v49 = v92;
    v92 = 0;
    v94 = v49;
    v93 = v91;
    sub_C45EE0((__int64)&v93, (__int64 *)a2);
    v50 = v94;
    v94 = 0;
    v96 = v50;
    v95 = v93;
    sub_C45EE0((__int64)&v95, a3);
    v51 = v96;
    v52 = v95;
    v90 = v96;
    v89 = (char *)v95;
    if ( v94 > 0x40 && v93 )
      j_j___libc_free_0_0(v93);
    if ( v92 > 0x40 && v91 )
      j_j___libc_free_0_0(v91);
    v53 = v88;
    if ( v88 > 0x40 )
      v54 = *(_QWORD *)(v87 + 8LL * ((v88 - 1) >> 6));
    else
      v54 = v87;
    v55 = v52;
    if ( v51 > 0x40 )
      v55 = *(_QWORD *)(v52 + 8LL * ((v51 - 1) >> 6));
    if ( ((v55 & (1LL << ((unsigned __int8)v51 - 1))) != 0) != ((v54 & (1LL << ((unsigned __int8)v88 - 1))) != 0) )
      goto LABEL_186;
    if ( v88 <= 0x40 )
      v56 = v87 == 0;
    else
      v56 = v53 == (unsigned int)sub_C444A0((__int64)&v87);
    v57 = v52 == 0;
    if ( v51 > 0x40 )
      v57 = v51 == (unsigned int)sub_C444A0((__int64)&v89);
    if ( v57 == v56 )
    {
      *(_BYTE *)(a1 + 16) = 0;
    }
    else
    {
LABEL_186:
      sub_C46A40((__int64)&v83, 1);
      v58 = v84;
      v84 = 0;
      *(_DWORD *)(a1 + 8) = v58;
      v59 = v83;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v59;
    }
    if ( v51 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v88 > 0x40 && v87 )
      j_j___libc_free_0_0(v87);
    v31 = v86;
  }
  if ( v31 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  if ( v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  if ( v82 > 0x40 && v81 )
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
  return a1;
}
