// Function: sub_23D2CB0
// Address: 0x23d2cb0
//
__int64 __fastcall sub_23D2CB0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  unsigned int v3; // ebx
  unsigned __int64 v4; // r13
  bool v5; // r12
  __int64 **v6; // r13
  unsigned int v7; // ebx
  unsigned __int64 v8; // rax
  int v9; // eax
  unsigned int v10; // esi
  int v11; // edx
  int v12; // eax
  __int64 v13; // rbx
  int v14; // edx
  __int64 v15; // rcx
  int v16; // eax
  int v17; // edx
  bool v18; // of
  __int64 v19; // rbx
  __int64 v20; // rax
  int v21; // edx
  int v22; // r14d
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // edx
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  int v28; // edx
  signed __int64 v29; // rbx
  unsigned int v30; // eax
  unsigned __int64 v31; // r14
  unsigned int v32; // eax
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rdx
  unsigned int v35; // eax
  __int64 v36; // rax
  _BYTE *v37; // r14
  __int64 (__fastcall *v38)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v39; // r12
  int v40; // r13d
  unsigned __int64 *v41; // rbx
  char *v42; // r13
  __int64 v43; // rdx
  unsigned int v44; // esi
  unsigned __int64 v45; // rax
  __int128 v46; // [rsp-8h] [rbp-1E8h]
  __int128 v47; // [rsp-8h] [rbp-1E8h]
  __int128 v48; // [rsp-8h] [rbp-1E8h]
  signed __int64 v49; // [rsp+10h] [rbp-1D0h]
  __int64 v50; // [rsp+38h] [rbp-1A8h]
  int v51; // [rsp+40h] [rbp-1A0h]
  int v52; // [rsp+40h] [rbp-1A0h]
  unsigned __int64 v53; // [rsp+40h] [rbp-1A0h]
  __int64 *v54; // [rsp+50h] [rbp-190h]
  unsigned int v55; // [rsp+50h] [rbp-190h]
  unsigned __int64 v57; // [rsp+68h] [rbp-178h] BYREF
  const void **v58; // [rsp+70h] [rbp-170h] BYREF
  const void **v59; // [rsp+78h] [rbp-168h] BYREF
  unsigned __int64 v60; // [rsp+80h] [rbp-160h] BYREF
  unsigned int v61; // [rsp+88h] [rbp-158h]
  unsigned __int64 v62; // [rsp+90h] [rbp-150h] BYREF
  unsigned int v63; // [rsp+98h] [rbp-148h]
  unsigned __int64 v64; // [rsp+A0h] [rbp-140h] BYREF
  unsigned int v65; // [rsp+A8h] [rbp-138h]
  unsigned __int64 v66; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v67; // [rsp+B8h] [rbp-128h]
  __int16 v68; // [rsp+D0h] [rbp-110h]
  unsigned __int64 v69; // [rsp+E0h] [rbp-100h] BYREF
  const void ***v70; // [rsp+E8h] [rbp-F8h]
  char v71; // [rsp+F0h] [rbp-F0h]
  const void ***v72; // [rsp+F8h] [rbp-E8h]
  __int16 v73; // [rsp+100h] [rbp-E0h]
  unsigned __int64 *v74; // [rsp+110h] [rbp-D0h] BYREF
  const void ***v75; // [rsp+118h] [rbp-C8h]
  char v76; // [rsp+120h] [rbp-C0h] BYREF
  const void ***v77; // [rsp+128h] [rbp-B8h]
  char v78; // [rsp+130h] [rbp-B0h]
  _BYTE v79[16]; // [rsp+138h] [rbp-A8h] BYREF
  __int64 v80; // [rsp+148h] [rbp-98h]
  __int64 v81; // [rsp+150h] [rbp-90h]
  __int64 *v82; // [rsp+158h] [rbp-88h]
  __int64 v83; // [rsp+160h] [rbp-80h]
  __int64 v84; // [rsp+168h] [rbp-78h] BYREF
  __int64 v85; // [rsp+170h] [rbp-70h]
  int v86; // [rsp+178h] [rbp-68h]
  void *v87; // [rsp+190h] [rbp-50h]

  v69 = (unsigned __int64)&v57;
  v70 = &v58;
  v71 = 0;
  v72 = &v59;
  LOBYTE(v73) = 0;
  if ( !(unsigned __int8)sub_23D2730((__int64)&v69, (char *)a1) )
  {
    v75 = &v59;
    v74 = &v57;
    v76 = 0;
    v77 = &v58;
    v78 = 0;
    result = sub_23D2B00((__int64)&v74, (char *)a1);
    if ( !(_BYTE)result )
      return result;
  }
  v61 = *((_DWORD *)v58 + 2);
  if ( v61 > 0x40 )
    sub_C43780((__int64)&v60, v58);
  else
    v60 = (unsigned __int64)*v58;
  sub_C46A40((__int64)&v60, 1);
  v3 = v61;
  v4 = v60;
  v61 = 0;
  v63 = v3;
  v62 = v60;
  if ( v3 > 0x40 )
  {
    if ( (unsigned int)sub_C44630((__int64)&v62) != 1 )
    {
LABEL_8:
      v5 = 1;
      goto LABEL_9;
    }
  }
  else if ( !v60 || (v60 & (v60 - 1)) != 0 )
  {
    goto LABEL_8;
  }
  LODWORD(v70) = *((_DWORD *)v58 + 2);
  if ( (unsigned int)v70 > 0x40 )
    sub_C43780((__int64)&v69, v58);
  else
    v69 = (unsigned __int64)*v58;
  sub_C46A40((__int64)&v69, 1);
  v30 = (unsigned int)v70;
  v31 = v69;
  LODWORD(v70) = 0;
  v55 = v30;
  LODWORD(v75) = v30;
  v74 = (unsigned __int64 *)v69;
  v32 = *((_DWORD *)v59 + 2);
  v65 = v32;
  if ( v32 <= 0x40 )
  {
    v33 = (unsigned __int64)*v59;
LABEL_64:
    v34 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v32) & ~v33;
    if ( !v32 )
      v34 = 0;
    v64 = v34;
    goto LABEL_67;
  }
  sub_C43780((__int64)&v64, v59);
  v32 = v65;
  if ( v65 <= 0x40 )
  {
    v33 = v64;
    goto LABEL_64;
  }
  sub_C43D10((__int64)&v64);
LABEL_67:
  sub_C46250((__int64)&v64);
  v35 = v65;
  v65 = 0;
  LODWORD(v67) = v35;
  v66 = v64;
  v5 = v31 != v64;
  if ( v35 > 0x40 )
  {
    v53 = v64;
    v5 = !sub_C43C50((__int64)&v66, (const void **)&v74);
    if ( v53 )
    {
      j_j___libc_free_0_0(v53);
      if ( v65 > 0x40 )
      {
        if ( v64 )
          j_j___libc_free_0_0(v64);
      }
    }
  }
  if ( v55 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( (unsigned int)v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
LABEL_9:
  if ( v3 > 0x40 && v4 )
    j_j___libc_free_0_0(v4);
  if ( v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( v5 )
    return 0;
  v6 = *(__int64 ***)(a1 + 8);
  v50 = *(_QWORD *)(v57 + 8);
  LODWORD(v70) = *((_DWORD *)v58 + 2);
  if ( (unsigned int)v70 > 0x40 )
    sub_C43780((__int64)&v69, v58);
  else
    v69 = (unsigned __int64)*v58;
  sub_C46A40((__int64)&v69, 1);
  v7 = (unsigned int)v70;
  LODWORD(v70) = 0;
  LODWORD(v75) = v7;
  v74 = (unsigned __int64 *)v69;
  if ( v7 > 0x40 )
  {
    if ( (unsigned int)sub_C44630((__int64)&v74) != 1 )
    {
      v10 = 0;
      goto LABEL_25;
    }
    v9 = sub_C444A0((__int64)&v74);
  }
  else
  {
    if ( !v69 || (v69 & (v69 - 1)) != 0 )
    {
      v10 = 0;
      goto LABEL_25;
    }
    _BitScanReverse64(&v8, v69);
    v9 = v7 + (v8 ^ 0x3F) - 64;
  }
  v10 = v7 - v9;
LABEL_25:
  v54 = (__int64 *)sub_BCCE00(*v6, v10);
  if ( (unsigned int)v75 > 0x40 && v74 )
    j_j___libc_free_0_0((unsigned __int64)v74);
  if ( (unsigned int)v70 > 0x40 && v69 )
    j_j___libc_free_0_0(v69);
  v11 = *((unsigned __int8 *)v6 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    v12 = *((_DWORD *)v6 + 8);
    BYTE4(v64) = (_BYTE)v11 == 18;
    LODWORD(v64) = v12;
    v54 = (__int64 *)sub_BCE1B0(v54, v64);
  }
  v66 = v57;
  *((_QWORD *)&v46 + 1) = 1;
  *(_QWORD *)&v46 = 0;
  v69 = v50;
  sub_DF8E30((__int64)&v74, 175, (__int64)v54, (char *)&v66, 1, 0, (char *)&v69, 1, 0, v46, 0);
  v13 = sub_DFD690((__int64)a2, (__int64)&v74);
  v51 = v14;
  if ( v82 != &v84 )
    _libc_free((unsigned __int64)v82);
  if ( v77 != (const void ***)v79 )
    _libc_free((unsigned __int64)v77);
  v15 = sub_DFD060(a2, 40, (__int64)v6, (__int64)v54);
  v16 = 1;
  if ( v17 != 1 )
    v16 = v51;
  v18 = __OFADD__(v15, v13);
  v19 = v15 + v13;
  v52 = v16;
  if ( v18 )
  {
    v45 = 0x8000000000000000LL;
    if ( v15 > 0 )
      v45 = 0x7FFFFFFFFFFFFFFFLL;
    v49 = v45;
  }
  else
  {
    v49 = v19;
  }
  v20 = sub_DFD060(a2, 42, (__int64)v6, v50);
  v22 = v21;
  v23 = v20;
  v69 = (unsigned __int64)v6;
  *((_QWORD *)&v47 + 1) = 1;
  *(_QWORD *)&v47 = 0;
  sub_DF8CB0((__int64)&v74, 330, (__int64)v6, (char *)&v69, 1, 0, 0, v47);
  v24 = sub_DFD690((__int64)a2, (__int64)&v74);
  if ( v25 == 1 )
    v22 = 1;
  v18 = __OFADD__(v24, v23);
  v26 = v24 + v23;
  if ( v18 )
  {
    v26 = 0x8000000000000000LL;
    if ( v24 > 0 )
      v26 = 0x7FFFFFFFFFFFFFFFLL;
  }
  if ( v82 != &v84 )
    _libc_free((unsigned __int64)v82);
  if ( v77 != (const void ***)v79 )
    _libc_free((unsigned __int64)v77);
  v69 = (unsigned __int64)v6;
  *((_QWORD *)&v48 + 1) = 1;
  *(_QWORD *)&v48 = 0;
  sub_DF8CB0((__int64)&v74, 329, (__int64)v6, (char *)&v69, 1, 0, 0, v48);
  v27 = sub_DFD690((__int64)a2, (__int64)&v74);
  if ( v28 == 1 )
    v22 = 1;
  v18 = __OFADD__(v27, v26);
  v29 = v27 + v26;
  if ( v18 )
  {
    v29 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v27 <= 0 )
      v29 = 0x8000000000000000LL;
  }
  if ( v82 != &v84 )
    _libc_free((unsigned __int64)v82);
  if ( v77 != (const void ***)v79 )
    _libc_free((unsigned __int64)v77);
  if ( v22 == v52 )
  {
    if ( v29 <= v49 )
      return 0;
  }
  else if ( v52 >= v22 )
  {
    return 0;
  }
  sub_23D0AB0((__int64)&v74, a1, 0, 0, 0);
  v73 = 257;
  HIDWORD(v62) = 0;
  v66 = (unsigned __int64)v54;
  v67 = v50;
  v36 = sub_B33D10((__int64)&v74, 0xAFu, (__int64)&v66, 2, (int)&v57, 1, (unsigned int)v62, (__int64)&v69);
  v68 = 257;
  v37 = (_BYTE *)v36;
  if ( v6 == *(__int64 ***)(v36 + 8) )
  {
    v39 = v36;
    goto LABEL_86;
  }
  v38 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v83 + 120LL);
  if ( v38 == sub_920130 )
  {
    if ( *v37 > 0x15u )
    {
LABEL_95:
      v73 = 257;
      v39 = sub_B51D30(40, (__int64)v37, (__int64)v6, (__int64)&v69, 0, 0);
      if ( (unsigned __int8)sub_920620(v39) )
      {
        v40 = v86;
        if ( v85 )
          sub_B99FD0(v39, 3u, v85);
        sub_B45150(v39, v40);
      }
      (*(void (__fastcall **)(__int64, __int64, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v84 + 16LL))(
        v84,
        v39,
        &v66,
        v80,
        v81);
      v41 = v74;
      v42 = (char *)&v74[2 * (unsigned int)v75];
      if ( v74 != (unsigned __int64 *)v42 )
      {
        do
        {
          v43 = v41[1];
          v44 = *(_DWORD *)v41;
          v41 += 2;
          sub_B99FD0(v39, v44, v43);
        }
        while ( v42 != (char *)v41 );
      }
      goto LABEL_86;
    }
    if ( (unsigned __int8)sub_AC4810(0x28u) )
      v39 = sub_ADAB70(40, (unsigned __int64)v37, v6, 0);
    else
      v39 = sub_AA93C0(0x28u, (unsigned __int64)v37, (__int64)v6);
  }
  else
  {
    v39 = v38(v83, 40u, v37, (__int64)v6);
  }
  if ( !v39 )
    goto LABEL_95;
LABEL_86:
  sub_BD84D0(a1, v39);
  nullsub_61();
  v87 = &unk_49DA100;
  nullsub_63();
  if ( v74 != (unsigned __int64 *)&v76 )
    _libc_free((unsigned __int64)v74);
  return 1;
}
