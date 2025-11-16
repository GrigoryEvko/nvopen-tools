// Function: sub_2A41510
// Address: 0x2a41510
//
__int64 *__fastcall sub_2A41510(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        const void *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        unsigned __int64 a12,
        char a13)
{
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // r12
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 *v24; // r14
  __int64 v25; // rax
  __int64 **v26; // rax
  unsigned __int8 *v27; // rbx
  __int64 (__fastcall *v28)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v29; // r10
  _QWORD *v30; // rax
  __int64 v31; // r9
  __int64 v32; // r14
  __int64 v33; // rbx
  unsigned int *v34; // rbx
  unsigned int *v35; // r12
  __int64 v36; // rdx
  unsigned int v37; // esi
  unsigned __int64 v38; // rsi
  int v39; // eax
  __int64 v40; // rsi
  __int64 *v42; // rax
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  int v45; // edx
  _QWORD *v46; // rax
  __int64 v47; // rbx
  unsigned int *v48; // r13
  unsigned int *v49; // r12
  __int64 v50; // rdx
  unsigned int v51; // esi
  _QWORD *v52; // rax
  _QWORD *v53; // r10
  _QWORD **v54; // rdx
  int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // rax
  unsigned int *v58; // r12
  unsigned int *v59; // r14
  __int64 v60; // rbx
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // [rsp+0h] [rbp-1B0h]
  __int64 v64; // [rsp+0h] [rbp-1B0h]
  __int64 v65; // [rsp+0h] [rbp-1B0h]
  __int64 v66; // [rsp+0h] [rbp-1B0h]
  __int64 v67; // [rsp+10h] [rbp-1A0h]
  __int64 v68; // [rsp+20h] [rbp-190h]
  unsigned __int64 v69; // [rsp+60h] [rbp-150h]
  __int64 v70; // [rsp+70h] [rbp-140h]
  __int64 v71; // [rsp+78h] [rbp-138h]
  __int64 v72; // [rsp+88h] [rbp-128h]
  char v73[32]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v74; // [rsp+B0h] [rbp-100h]
  const char *v75[4]; // [rsp+C0h] [rbp-F0h] BYREF
  __int16 v76; // [rsp+E0h] [rbp-D0h]
  unsigned int *v77; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+F8h] [rbp-B8h]
  _BYTE v79[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+120h] [rbp-90h]
  __int64 v81; // [rsp+128h] [rbp-88h]
  __int64 v82; // [rsp+130h] [rbp-80h]
  __int64 *v83; // [rsp+138h] [rbp-78h]
  void **v84; // [rsp+140h] [rbp-70h]
  void **v85; // [rsp+148h] [rbp-68h]
  __int64 v86; // [rsp+150h] [rbp-60h]
  int v87; // [rsp+158h] [rbp-58h]
  __int16 v88; // [rsp+15Ch] [rbp-54h]
  char v89; // [rsp+15Eh] [rbp-52h]
  __int64 v90; // [rsp+160h] [rbp-50h]
  __int64 v91; // [rsp+168h] [rbp-48h]
  void *v92; // [rsp+170h] [rbp-40h] BYREF
  void *v93; // [rsp+178h] [rbp-38h] BYREF

  v16 = sub_2A3F100(a2, a5, a6, a7, a8, a13);
  v71 = v17;
  v69 = v16;
  v89 = 7;
  v18 = sub_2A41400((__int64)a2, a3, a4);
  v19 = *a2;
  v86 = 0;
  v87 = 0;
  v83 = v19;
  v84 = &v92;
  v85 = &v93;
  v88 = 512;
  v90 = 0;
  v92 = &unk_49DA100;
  v91 = 0;
  LOWORD(v82) = 0;
  v93 = &unk_49DA0B0;
  v20 = *(_QWORD *)(v18 + 80);
  v77 = (unsigned int *)v79;
  v78 = 0x200000000LL;
  if ( v20 )
    v20 -= 24;
  v80 = 0;
  v81 = 0;
  v70 = v18;
  if ( a13 )
  {
    v75[0] = "ret";
    v76 = 259;
    sub_BD6B50((unsigned __int8 *)v20, v75);
    v21 = *a2;
    v75[0] = "entry";
    v76 = 259;
    v22 = sub_22077B0(0x50u);
    v23 = v22;
    if ( v22 )
      sub_AA4D50(v22, (__int64)v21, (__int64)v75, v70, v20);
    v24 = *a2;
    v75[0] = "callfunc";
    v76 = 259;
    v25 = sub_22077B0(0x50u);
    v68 = v25;
    if ( v25 )
      sub_AA4D50(v25, (__int64)v24, (__int64)v75, v70, v20);
    v26 = (__int64 **)sub_BCE3C0(*a2, *(_DWORD *)(*(_QWORD *)(v71 + 8) + 8LL) >> 8);
    v80 = v23;
    v81 = v23 + 48;
    LOWORD(v82) = 0;
    v74 = 257;
    v27 = (unsigned __int8 *)sub_AC9EC0(v26);
    v28 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v84 + 7);
    if ( v28 == sub_928890 )
    {
      if ( *(_BYTE *)v71 > 0x15u || *v27 > 0x15u )
        goto LABEL_36;
      v29 = sub_AAB310(0x21u, (unsigned __int8 *)v71, v27);
    }
    else
    {
      v29 = v28((__int64)v84, 33u, (_BYTE *)v71, v27);
    }
    if ( v29 )
    {
LABEL_13:
      v67 = v29;
      v76 = 257;
      v30 = sub_BD2C40(72, 3u);
      v32 = (__int64)v30;
      if ( v30 )
        sub_B4C9A0((__int64)v30, v68, v20, v67, 3u, v31, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v85 + 2))(v85, v32, v75, v81, v82);
      v33 = 4LL * (unsigned int)v78;
      if ( v77 != &v77[v33] )
      {
        v63 = v20;
        v34 = &v77[v33];
        v35 = v77;
        do
        {
          v36 = *((_QWORD *)v35 + 1);
          v37 = *v35;
          v35 += 4;
          sub_B99FD0(v32, v37, v36);
        }
        while ( v34 != v35 );
        v20 = v63;
      }
      LOWORD(v82) = 0;
      v80 = v68;
      v81 = v68 + 48;
      goto LABEL_25;
    }
LABEL_36:
    v76 = 257;
    v52 = sub_BD2C40(72, unk_3F10FD0);
    v53 = v52;
    if ( v52 )
    {
      v64 = (__int64)v52;
      v54 = *(_QWORD ***)(v71 + 8);
      v55 = *((unsigned __int8 *)v54 + 8);
      if ( (unsigned int)(v55 - 17) > 1 )
      {
        v57 = sub_BCB2A0(*v54);
      }
      else
      {
        BYTE4(v72) = (_BYTE)v55 == 18;
        LODWORD(v72) = *((_DWORD *)v54 + 8);
        v56 = (__int64 *)sub_BCB2A0(*v54);
        v57 = sub_BCE1B0(v56, v72);
      }
      sub_B523C0(v64, v57, 53, 33, v71, (__int64)v27, (__int64)v75, 0, 0, 0);
      v53 = (_QWORD *)v64;
    }
    v65 = (__int64)v53;
    (*((void (__fastcall **)(void **, _QWORD *, char *, __int64, __int64))*v85 + 2))(v85, v53, v73, v81, v82);
    v29 = v65;
    if ( v77 != &v77[4 * (unsigned int)v78] )
    {
      v66 = v20;
      v58 = v77;
      v59 = &v77[4 * (unsigned int)v78];
      v60 = v29;
      do
      {
        v61 = *((_QWORD *)v58 + 1);
        v62 = *v58;
        v58 += 4;
        sub_B99FD0(v60, v62, v61);
      }
      while ( v59 != v58 );
      v20 = v66;
      v29 = v60;
    }
    goto LABEL_13;
  }
  v38 = *(_QWORD *)(v20 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v38 == v20 + 48 )
  {
    v40 = 0;
  }
  else
  {
    if ( !v38 )
      BUG();
    v39 = *(unsigned __int8 *)(v38 - 24);
    v40 = v38 - 24;
    if ( (unsigned int)(v39 - 30) >= 0xB )
      v40 = 0;
  }
  sub_D5F1F0((__int64)&v77, v40);
LABEL_25:
  v76 = 257;
  sub_921880(&v77, v69, v71, a9, a10, (__int64)v75, 0);
  if ( a12 )
  {
    v42 = (__int64 *)sub_BCB120(v83);
    v43 = sub_BCF480(v42, 0, 0, 0);
    v44 = sub_BA8C10((__int64)a2, a11, a12, v43, 0);
    v76 = 257;
    sub_921880(&v77, v44, v45, 0, 0, (__int64)v75, 0);
    if ( !a13 )
      goto LABEL_27;
  }
  else if ( !a13 )
  {
    goto LABEL_27;
  }
  v76 = 257;
  v46 = sub_BD2C40(72, 1u);
  v47 = (__int64)v46;
  if ( v46 )
    sub_B4C8F0((__int64)v46, v20, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v85 + 2))(v85, v47, v75, v81, v82);
  v48 = v77;
  v49 = &v77[4 * (unsigned int)v78];
  if ( v77 != v49 )
  {
    do
    {
      v50 = *((_QWORD *)v48 + 1);
      v51 = *v48;
      v48 += 4;
      sub_B99FD0(v47, v51, v50);
    }
    while ( v49 != v48 );
  }
LABEL_27:
  *a1 = v70;
  a1[1] = v69;
  a1[2] = v71;
  nullsub_61();
  v92 = &unk_49DA100;
  nullsub_63();
  if ( v77 != (unsigned int *)v79 )
    _libc_free((unsigned __int64)v77);
  return a1;
}
