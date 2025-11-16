// Function: sub_2A35A50
// Address: 0x2a35a50
//
__int64 __fastcall sub_2A35A50(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        unsigned __int8 *a4,
        const char *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 v12; // rax
  __int64 v13; // rbx
  _QWORD *v14; // r15
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // r13
  const char *v25; // r15
  unsigned __int16 v26; // bx
  _QWORD *v27; // rdi
  const char *v28; // r15
  unsigned __int16 v29; // bx
  _QWORD *v30; // rdi
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // rbx
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // rbx
  int v37; // eax
  int v38; // eax
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v45; // rbx
  __int64 v46; // rdi
  __int64 (__fastcall *v47)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v48; // r11
  _QWORD *v49; // rdi
  __int64 v50; // r8
  __int64 v51; // r9
  int v52; // eax
  int v53; // eax
  unsigned int v54; // edx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rdx
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v64; // r13
  __int64 v65; // r12
  __int64 v66; // rdx
  unsigned int v67; // esi
  _QWORD *v68; // rax
  _QWORD *v69; // r11
  _QWORD **v70; // rdx
  int v71; // ecx
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 v74; // rbx
  __int64 v75; // r15
  __int64 v76; // r12
  __int64 v77; // rdx
  unsigned int v78; // esi
  __int64 v81; // [rsp+10h] [rbp-110h]
  __int64 v82; // [rsp+18h] [rbp-108h]
  __int64 v83; // [rsp+18h] [rbp-108h]
  __int64 v84; // [rsp+18h] [rbp-108h]
  __int64 v85; // [rsp+18h] [rbp-108h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  const char *v89; // [rsp+30h] [rbp-F0h]
  __int64 v90; // [rsp+30h] [rbp-F0h]
  __int64 v91; // [rsp+30h] [rbp-F0h]
  __int64 v92; // [rsp+30h] [rbp-F0h]
  __int64 v93; // [rsp+38h] [rbp-E8h]
  __int64 v95; // [rsp+40h] [rbp-E0h]
  __int64 v96; // [rsp+40h] [rbp-E0h]
  __int64 v98; // [rsp+58h] [rbp-C8h]
  const char *v99; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v100; // [rsp+68h] [rbp-B8h]
  char *v101; // [rsp+70h] [rbp-B0h]
  __int16 v102; // [rsp+80h] [rbp-A0h]
  const char *v103; // [rsp+90h] [rbp-90h] BYREF
  __int64 v104; // [rsp+98h] [rbp-88h]
  char *v105; // [rsp+A0h] [rbp-80h]
  unsigned __int64 v106; // [rsp+A8h] [rbp-78h]
  __int64 v107; // [rsp+B0h] [rbp-70h]
  unsigned __int64 v108; // [rsp+B8h] [rbp-68h]
  __int64 v109; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v110; // [rsp+C8h] [rbp-58h]
  __int64 v111; // [rsp+D0h] [rbp-50h]
  unsigned __int64 v112; // [rsp+D8h] [rbp-48h]
  __int64 v113; // [rsp+E0h] [rbp-40h]
  unsigned __int64 v114; // [rsp+E8h] [rbp-38h]

  v12 = sub_AA48A0(a1);
  v13 = *(_QWORD *)(a1 + 72);
  v103 = a5;
  v14 = (_QWORD *)v12;
  v104 = a6;
  LOWORD(v107) = 773;
  v105 = ".header";
  v15 = sub_AA48A0(a1);
  v16 = sub_22077B0(0x50u);
  v17 = v16;
  if ( v16 )
    sub_AA4D50(v16, v15, (__int64)&v103, v13, a2);
  v18 = *(_QWORD *)(v17 + 72);
  LOWORD(v107) = 773;
  v103 = a5;
  v104 = a6;
  v105 = ".body";
  v19 = sub_AA48A0(v17);
  v20 = sub_22077B0(0x50u);
  v87 = v20;
  if ( v20 )
    sub_AA4D50(v20, v19, (__int64)&v103, v18, a2);
  v21 = *(_QWORD *)(v17 + 72);
  LOWORD(v107) = 773;
  v103 = a5;
  v82 = v21;
  v104 = a6;
  v105 = ".latch";
  v22 = sub_AA48A0(v17);
  v23 = sub_22077B0(0x50u);
  v24 = v23;
  if ( v23 )
    sub_AA4D50(v23, v22, (__int64)&v103, v82, a2);
  v83 = sub_BCB2E0(v14);
  sub_B43C20((__int64)&v103, v17);
  v25 = v103;
  v26 = v104;
  v27 = sub_BD2C40(72, 1u);
  if ( v27 )
    sub_B4C8F0((__int64)v27, v87, 1u, (__int64)v25, v26);
  sub_B43C20((__int64)&v103, v87);
  v28 = v103;
  v29 = v104;
  v30 = sub_BD2C40(72, 1u);
  if ( v30 )
    sub_B4C8F0((__int64)v30, v24, 1u, (__int64)v28, v29);
  v31 = *(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v31 == v17 + 48 )
  {
    v32 = 0;
  }
  else
  {
    if ( !v31 )
      BUG();
    v32 = v31 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v31 - 24) - 30 >= 0xB )
      v32 = 0;
  }
  v33 = v32 + 24;
  LOWORD(v107) = 773;
  v103 = a5;
  v104 = a6;
  v105 = ".iv";
  v34 = sub_BD2DA0(80);
  v35 = v34;
  if ( v34 )
  {
    sub_B44260(v34, v83, 55, 0x8000000u, v33, 0);
    *(_DWORD *)(v35 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v35, &v103);
    sub_BD2A10(v35, *(_DWORD *)(v35 + 72), 1);
  }
  v36 = sub_AD64C0(v83, 0, 0);
  v37 = *(_DWORD *)(v35 + 4) & 0x7FFFFFF;
  if ( v37 == *(_DWORD *)(v35 + 72) )
  {
    sub_B48D90(v35);
    v37 = *(_DWORD *)(v35 + 4) & 0x7FFFFFF;
  }
  v38 = (v37 + 1) & 0x7FFFFFF;
  v39 = v38 | *(_DWORD *)(v35 + 4) & 0xF8000000;
  v40 = *(_QWORD *)(v35 - 8) + 32LL * (unsigned int)(v38 - 1);
  *(_DWORD *)(v35 + 4) = v39;
  if ( *(_QWORD *)v40 )
  {
    v41 = *(_QWORD *)(v40 + 8);
    **(_QWORD **)(v40 + 16) = v41;
    if ( v41 )
      *(_QWORD *)(v41 + 16) = *(_QWORD *)(v40 + 16);
  }
  *(_QWORD *)v40 = v36;
  if ( v36 )
  {
    v42 = *(_QWORD *)(v36 + 16);
    *(_QWORD *)(v40 + 8) = v42;
    if ( v42 )
      *(_QWORD *)(v42 + 16) = v40 + 8;
    *(_QWORD *)(v40 + 16) = v36 + 16;
    *(_QWORD *)(v36 + 16) = v40;
  }
  *(_QWORD *)(*(_QWORD *)(v35 - 8) + 32LL * *(unsigned int *)(v35 + 72)
                                   + 8LL * ((*(_DWORD *)(v35 + 4) & 0x7FFFFFFu) - 1)) = a1;
  v102 = 773;
  a7[6] = v24;
  a7[7] = v24 + 48;
  v43 = a7[10];
  *((_WORD *)a7 + 32) = 0;
  v99 = a5;
  v100 = a6;
  v101 = ".step";
  v44 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v43 + 32LL);
  if ( v44 != sub_9201A0 )
  {
    v45 = v44(v43, 13u, (_BYTE *)v35, a4, 0, 0);
    goto LABEL_31;
  }
  if ( *(_BYTE *)v35 <= 0x15u && *a4 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(13) )
      v45 = sub_AD5570(13, v35, a4, 0, 0);
    else
      v45 = sub_AABE40(0xDu, (unsigned __int8 *)v35, a4);
LABEL_31:
    if ( v45 )
      goto LABEL_32;
  }
  LOWORD(v107) = 257;
  v45 = sub_B504D0(13, v35, (__int64)a4, (__int64)&v103, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a7[11] + 16LL))(
    a7[11],
    v45,
    &v99,
    a7[7],
    a7[8]);
  if ( *a7 != *a7 + 16LL * *((unsigned int *)a7 + 2) )
  {
    v84 = v24;
    v64 = *a7;
    v81 = v17;
    v65 = *a7 + 16LL * *((unsigned int *)a7 + 2);
    do
    {
      v66 = *(_QWORD *)(v64 + 8);
      v67 = *(_DWORD *)v64;
      v64 += 16;
      sub_B99FD0(v45, v67, v66);
    }
    while ( v65 != v64 );
    v24 = v84;
    v17 = v81;
  }
LABEL_32:
  v102 = 773;
  v99 = a5;
  v100 = a6;
  v101 = ".cond";
  v46 = a7[10];
  v47 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v46 + 56LL);
  if ( v47 != sub_928890 )
  {
    v48 = v47(v46, 33u, (_BYTE *)v45, a3);
LABEL_36:
    if ( v48 )
      goto LABEL_37;
    goto LABEL_61;
  }
  if ( *(_BYTE *)v45 <= 0x15u && *a3 <= 0x15u )
  {
    v48 = sub_AAB310(0x21u, (unsigned __int8 *)v45, a3);
    goto LABEL_36;
  }
LABEL_61:
  LOWORD(v107) = 257;
  v68 = sub_BD2C40(72, unk_3F10FD0);
  v69 = v68;
  if ( v68 )
  {
    v70 = *(_QWORD ***)(v45 + 8);
    v90 = (__int64)v68;
    v71 = *((unsigned __int8 *)v70 + 8);
    if ( (unsigned int)(v71 - 17) > 1 )
    {
      v73 = sub_BCB2A0(*v70);
    }
    else
    {
      BYTE4(v98) = (_BYTE)v71 == 18;
      LODWORD(v98) = *((_DWORD *)v70 + 8);
      v72 = (__int64 *)sub_BCB2A0(*v70);
      v73 = sub_BCE1B0(v72, v98);
    }
    sub_B523C0(v90, v73, 53, 33, v45, (__int64)a3, (__int64)&v103, 0, 0, 0);
    v69 = (_QWORD *)v90;
  }
  v91 = (__int64)v69;
  (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)a7[11] + 16LL))(
    a7[11],
    v69,
    &v99,
    a7[7],
    a7[8]);
  v48 = v91;
  if ( *a7 != *a7 + 16LL * *((unsigned int *)a7 + 2) )
  {
    v92 = v45;
    v74 = v48;
    v96 = v35;
    v75 = *a7 + 16LL * *((unsigned int *)a7 + 2);
    v85 = v17;
    v76 = *a7;
    do
    {
      v77 = *(_QWORD *)(v76 + 8);
      v78 = *(_DWORD *)v76;
      v76 += 16;
      sub_B99FD0(v74, v78, v77);
    }
    while ( v75 != v76 );
    v48 = v74;
    v35 = v96;
    v45 = v92;
    v17 = v85;
  }
LABEL_37:
  v95 = v48;
  sub_B43C20((__int64)&v103, v24);
  v89 = v103;
  v93 = (unsigned __int16)v104;
  v49 = sub_BD2C40(72, 3u);
  if ( v49 )
    sub_B4C9A0((__int64)v49, v17, a2, v95, 3u, v93, (__int64)v89, v93);
  v52 = *(_DWORD *)(v35 + 4) & 0x7FFFFFF;
  if ( v52 == *(_DWORD *)(v35 + 72) )
  {
    sub_B48D90(v35);
    v52 = *(_DWORD *)(v35 + 4) & 0x7FFFFFF;
  }
  v53 = (v52 + 1) & 0x7FFFFFF;
  v54 = v53 | *(_DWORD *)(v35 + 4) & 0xF8000000;
  v55 = *(_QWORD *)(v35 - 8) + 32LL * (unsigned int)(v53 - 1);
  *(_DWORD *)(v35 + 4) = v54;
  if ( *(_QWORD *)v55 )
  {
    v56 = *(_QWORD *)(v55 + 8);
    **(_QWORD **)(v55 + 16) = v56;
    if ( v56 )
      *(_QWORD *)(v56 + 16) = *(_QWORD *)(v55 + 16);
  }
  *(_QWORD *)v55 = v45;
  if ( v45 )
  {
    v57 = *(_QWORD *)(v45 + 16);
    *(_QWORD *)(v55 + 8) = v57;
    if ( v57 )
      *(_QWORD *)(v57 + 16) = v55 + 8;
    *(_QWORD *)(v55 + 16) = v45 + 16;
    *(_QWORD *)(v45 + 16) = v55;
  }
  *(_QWORD *)(*(_QWORD *)(v35 - 8) + 32LL * *(unsigned int *)(v35 + 72)
                                   + 8LL * ((*(_DWORD *)(v35 + 4) & 0x7FFFFFFu) - 1)) = v24;
  v58 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v58 == a1 + 48 )
    goto LABEL_76;
  if ( !v58 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v58 - 24) - 30 > 0xA )
LABEL_76:
    BUG();
  v59 = *(_QWORD *)(v58 - 56);
  if ( v59 )
  {
    v60 = *(_QWORD *)(v58 - 48);
    v61 = v59 | 4;
    **(_QWORD **)(v58 - 40) = v60;
    if ( v60 )
      *(_QWORD *)(v60 + 16) = *(_QWORD *)(v58 - 40);
  }
  else
  {
    v61 = 4;
  }
  *(_QWORD *)(v58 - 56) = v17;
  v62 = *(_QWORD *)(v17 + 16);
  *(_QWORD *)(v58 - 48) = v62;
  if ( v62 )
  {
    v50 = v58 - 48;
    *(_QWORD *)(v62 + 16) = v58 - 48;
  }
  *(_QWORD *)(v58 - 40) = v17 + 16;
  *(_QWORD *)(v17 + 16) = v58 - 56;
  v104 = v61;
  v103 = (const char *)a1;
  v106 = v87 & 0xFFFFFFFFFFFFFFFBLL;
  v112 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v108 = v24 & 0xFFFFFFFFFFFFFFFBLL;
  v113 = a1;
  v110 = v17 & 0xFFFFFFFFFFFFFFFBLL;
  v114 = v17 & 0xFFFFFFFFFFFFFFFBLL;
  v105 = (char *)v17;
  v107 = v87;
  v109 = v24;
  v111 = v24;
  sub_FFDB80(a8, (unsigned __int64 *)&v103, 6, a1, v50, v51);
  sub_D4F330(a9, v17, a10);
  sub_D4F330(a9, v87, a10);
  sub_D4F330(a9, v24, a10);
  return v87;
}
