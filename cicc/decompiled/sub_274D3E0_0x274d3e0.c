// Function: sub_274D3E0
// Address: 0x274d3e0
//
__int64 __fastcall sub_274D3E0(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rdx
  char v4; // r15
  unsigned int v5; // eax
  unsigned int v6; // r13d
  unsigned __int8 *v8; // r14
  unsigned int v9; // r14d
  unsigned int v10; // ebx
  unsigned int v11; // eax
  unsigned __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 *v14; // r14
  int v15; // edx
  const char *v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int8 *v19; // rbx
  const char *v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // r14
  const char *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r14
  const char *v27; // rax
  __int64 **v28; // rbx
  __int64 v29; // rdx
  __int64 (__fastcall *v30)(__int64, __int64, __int64, __int64 **, __int64, __int64); // rax
  __int64 v31; // r15
  bool v32; // al
  const char *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  const char *v36; // rax
  __int64 v37; // rdx
  __int64 (__fastcall *v38)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v39; // rax
  unsigned int *v40; // rbx
  unsigned int *v41; // r13
  __int64 v42; // rdx
  unsigned int v43; // esi
  _QWORD *v44; // rax
  unsigned int *v45; // rbx
  unsigned int *v46; // r13
  __int64 v47; // rdx
  unsigned int v48; // esi
  const char *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r14
  const char *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  _QWORD *v56; // rax
  __int64 v57; // r14
  __int64 v58; // rdx
  unsigned int *v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // r14
  unsigned int *v65; // rbx
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // [rsp-10h] [rbp-1A0h]
  unsigned __int8 *v69; // [rsp+0h] [rbp-190h]
  char v70; // [rsp+Eh] [rbp-182h]
  unsigned __int8 v71; // [rsp+Fh] [rbp-181h]
  __int64 **v72; // [rsp+10h] [rbp-180h]
  unsigned int *v73; // [rsp+10h] [rbp-180h]
  unsigned __int8 *v74; // [rsp+18h] [rbp-178h]
  unsigned int *v75; // [rsp+18h] [rbp-178h]
  __int64 v76; // [rsp+28h] [rbp-168h]
  unsigned __int64 v77; // [rsp+30h] [rbp-160h] BYREF
  unsigned int v78; // [rsp+38h] [rbp-158h]
  unsigned __int64 v79; // [rsp+40h] [rbp-150h]
  unsigned int v80; // [rsp+48h] [rbp-148h]
  unsigned __int64 v81; // [rsp+50h] [rbp-140h] BYREF
  unsigned int v82; // [rsp+58h] [rbp-138h]
  unsigned __int64 v83; // [rsp+60h] [rbp-130h]
  unsigned int v84; // [rsp+68h] [rbp-128h]
  const char *v85; // [rsp+70h] [rbp-120h] BYREF
  __int64 v86; // [rsp+78h] [rbp-118h]
  char *v87; // [rsp+80h] [rbp-110h]
  __int16 v88; // [rsp+90h] [rbp-100h]
  const char *v89; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v90; // [rsp+A8h] [rbp-E8h]
  char *v91; // [rsp+B0h] [rbp-E0h]
  __int16 v92; // [rsp+C0h] [rbp-D0h]
  unsigned int *v93; // [rsp+D0h] [rbp-C0h] BYREF
  unsigned int v94; // [rsp+D8h] [rbp-B8h]
  unsigned __int64 v95; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v96; // [rsp+E8h] [rbp-A8h]
  __int64 v97; // [rsp+108h] [rbp-88h]
  __int64 v98; // [rsp+110h] [rbp-80h]
  __int64 v99; // [rsp+120h] [rbp-70h]
  __int64 v100; // [rsp+128h] [rbp-68h]
  void *v101; // [rsp+150h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v2 = *(__int64 **)(a1 - 8);
  else
    v2 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  sub_22CEA30((__int64)&v77, a2, v2, 0);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  sub_22CEA30((__int64)&v81, a2, (__int64 *)(v3 + 32), 1);
  v4 = *(_BYTE *)a1;
  v74 = *(unsigned __int8 **)(a1 - 64);
  v72 = *(__int64 ***)(a1 + 8);
  v70 = *(_BYTE *)a1;
  v69 = *(unsigned __int8 **)(a1 - 32);
  LOBYTE(v5) = sub_ABB410((__int64 *)&v77, 36, (__int64 *)&v81);
  v6 = v5;
  if ( (_BYTE)v5 )
  {
    if ( v4 != 51 )
      v74 = (unsigned __int8 *)sub_AD6530((__int64)v72, 36);
    sub_BD84D0(a1, (__int64)v74);
    sub_B43D60((_QWORD *)a1);
    goto LABEL_9;
  }
  sub_AB9DC0((__int64)&v93, (__int64)&v81, (__int64)&v81);
  v71 = 0;
  if ( !sub_ABB410((__int64 *)&v77, 36, (__int64 *)&v93) )
    v71 = sub_AB06D0((__int64)&v81) ^ 1;
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  if ( v94 > 0x40 && v93 )
    j_j___libc_free_0_0((unsigned __int64)v93);
  if ( !v71 )
  {
    sub_23D0AB0((__int64)&v93, a1, 0, 0, 0);
    if ( sub_ABB410((__int64 *)&v77, 35, (__int64 *)&v81) )
    {
      if ( v70 == 51 )
      {
        v92 = 257;
        v8 = (unsigned __int8 *)sub_929DE0(&v93, v74, v69, (__int64)&v89, 1u, 0);
      }
      else
      {
        v8 = (unsigned __int8 *)sub_AD64C0(*(_QWORD *)(a1 + 8), 1, 0);
      }
      goto LABEL_36;
    }
    if ( v70 == 51 )
    {
      if ( !sub_98EF80(v74, 0, 0, 0, 0) )
      {
        v85 = sub_BD5D20((__int64)v74);
        v87 = ".frozen";
        v88 = 773;
        v86 = v62;
        v92 = 257;
        v63 = sub_BD2C40(72, unk_3F10A14);
        v64 = (__int64)v63;
        if ( v63 )
          sub_B549F0((__int64)v63, (__int64)v74, (__int64)&v89, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v100 + 16LL))(
          v100,
          v64,
          &v85,
          v97,
          v98);
        v65 = v93;
        v75 = &v93[4 * v94];
        if ( v93 != v75 )
        {
          do
          {
            v66 = *((_QWORD *)v65 + 1);
            v67 = *v65;
            v65 += 4;
            sub_B99FD0(v64, v67, v66);
          }
          while ( v75 != v65 );
        }
        v74 = (unsigned __int8 *)v64;
      }
      if ( !sub_98EF80(v69, 0, 0, 0, 0) )
      {
        v85 = sub_BD5D20((__int64)v69);
        v87 = ".frozen";
        v88 = 773;
        v86 = v55;
        v92 = 257;
        v56 = sub_BD2C40(72, unk_3F10A14);
        v57 = (__int64)v56;
        if ( v56 )
          sub_B549F0((__int64)v56, (__int64)v69, (__int64)&v89, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v100 + 16LL))(
          v100,
          v57,
          &v85,
          v97,
          v98);
        v58 = 4LL * v94;
        v59 = v93;
        v73 = &v93[v58];
        while ( v73 != v59 )
        {
          v60 = *((_QWORD *)v59 + 1);
          v61 = *v59;
          v59 += 4;
          sub_B99FD0(v57, v61, v60);
        }
        v69 = (unsigned __int8 *)v57;
      }
      v49 = sub_BD5D20(a1);
      v90 = v50;
      v92 = 773;
      v89 = v49;
      v91 = ".urem";
      v51 = sub_929DE0(&v93, v74, v69, (__int64)&v89, 1u, 0);
      v52 = sub_BD5D20(a1);
      v92 = 773;
      v89 = v52;
      v90 = v53;
      v91 = ".cmp";
      v54 = sub_92B530(&v93, 0x24u, (__int64)v74, v69, (__int64)&v89);
      v92 = 257;
      v8 = (unsigned __int8 *)sub_B36550(&v93, v54, (__int64)v74, v51, (__int64)&v89, 0);
      goto LABEL_36;
    }
    v33 = sub_BD5D20(a1);
    v90 = v34;
    v89 = v33;
    v92 = 773;
    v91 = ".cmp";
    v35 = sub_92B530(&v93, 0x23u, (__int64)v74, v69, (__int64)&v89);
    v36 = sub_BD5D20(a1);
    v88 = 773;
    v85 = v36;
    v87 = ".udiv";
    v86 = v37;
    if ( v72 == *(__int64 ***)(v35 + 8) )
    {
      v8 = (unsigned __int8 *)v35;
    }
    else
    {
      v38 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v99 + 120LL);
      if ( v38 == sub_920130 )
      {
        if ( *(_BYTE *)v35 > 0x15u )
        {
LABEL_66:
          v92 = 257;
          v39 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
          v8 = v39;
          if ( v39 )
            sub_B515B0((__int64)v39, v35, (__int64)v72, (__int64)&v89, 0, 0);
          (*(void (__fastcall **)(__int64, unsigned __int8 *, const char **, __int64, __int64))(*(_QWORD *)v100 + 16LL))(
            v100,
            v8,
            &v85,
            v97,
            v98);
          v40 = v93;
          v41 = &v93[4 * v94];
          if ( v93 != v41 )
          {
            do
            {
              v42 = *((_QWORD *)v40 + 1);
              v43 = *v40;
              v40 += 4;
              sub_B99FD0((__int64)v8, v43, v42);
            }
            while ( v41 != v40 );
          }
          goto LABEL_36;
        }
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v8 = (unsigned __int8 *)sub_ADAB70(39, v35, v72, 0);
        else
          v8 = (unsigned __int8 *)sub_AA93C0(0x27u, v35, (__int64)v72);
      }
      else
      {
        v8 = (unsigned __int8 *)v38(v99, 39u, (_BYTE *)v35, (__int64)v72);
      }
      if ( !v8 )
        goto LABEL_66;
    }
LABEL_36:
    sub_BD6B90(v8, (unsigned __int8 *)a1);
    sub_BD84D0(a1, (__int64)v8);
    sub_B43D60((_QWORD *)a1);
    nullsub_61();
    v101 = &unk_49DA100;
    nullsub_63();
    if ( v93 != (unsigned int *)&v95 )
      _libc_free((unsigned __int64)v93);
    v6 = 1;
    goto LABEL_9;
  }
  v9 = 8;
  v10 = sub_AB1CA0((__int64)&v81);
  v11 = sub_AB1CA0((__int64)&v77);
  if ( v10 < v11 )
    v10 = v11;
  if ( v10 )
  {
    v12 = v10 - 1LL;
    if ( v12 )
    {
      _BitScanReverse64(&v12, v12);
      if ( (unsigned int)(1LL << (64 - ((unsigned __int8)v12 ^ 0x3Fu))) >= 8 )
        v9 = 1LL << (64 - ((unsigned __int8)v12 ^ 0x3Fu));
    }
  }
  if ( v9 < (unsigned int)sub_BCB060(*(_QWORD *)(a1 + 8)) )
  {
    sub_23D0AB0((__int64)&v93, a1, 0, 0, 0);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = (__int64 *)sub_BCD140(*(_QWORD **)v13, v9);
    v15 = *(unsigned __int8 *)(v13 + 8);
    if ( (unsigned int)(v15 - 17) <= 1 )
    {
      BYTE4(v76) = (_BYTE)v15 == 18;
      LODWORD(v76) = *(_DWORD *)(v13 + 32);
      v14 = (__int64 *)sub_BCE1B0(v14, v76);
    }
    v16 = sub_BD5D20(a1);
    v17 = *(_QWORD *)(a1 - 64);
    v92 = 773;
    v89 = v16;
    v90 = v18;
    v91 = ".lhs.trunc";
    v19 = (unsigned __int8 *)sub_274BD20((__int64 *)&v93, v17, (__int64 **)v14, (__int64)&v89);
    v20 = sub_BD5D20(a1);
    v92 = 773;
    v21 = *(_QWORD *)(a1 - 32);
    v89 = v20;
    v90 = v22;
    v91 = ".rhs.trunc";
    v23 = (unsigned __int8 *)sub_274BD20((__int64 *)&v93, v21, (__int64 **)v14, (__int64)&v89);
    v24 = sub_BD5D20(a1);
    LOBYTE(v21) = *(_BYTE *)a1;
    v92 = 261;
    v90 = v25;
    v89 = v24;
    v26 = sub_274BB90((__int64 *)&v93, (unsigned __int8)v21 - 29, v19, v23, (int)v85, 0, (__int64)&v89, 0);
    v27 = sub_BD5D20(a1);
    v28 = *(__int64 ***)(a1 + 8);
    v85 = v27;
    v88 = 773;
    v86 = v29;
    v87 = ".zext";
    if ( v28 == *(__int64 ***)(v26 + 8) )
    {
      v31 = v26;
      goto LABEL_54;
    }
    v30 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 **, __int64, __int64))(*(_QWORD *)v99 + 120LL);
    if ( (char *)v30 == (char *)sub_920130 )
    {
      if ( *(_BYTE *)v26 > 0x15u )
        goto LABEL_71;
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v31 = sub_ADAB70(39, v26, v28, 0);
      else
        v31 = sub_AA93C0(0x27u, v26, (__int64)v28);
    }
    else
    {
      v31 = v30(v99, 39, v26, v28, 773, v68);
    }
    if ( v31 )
    {
LABEL_54:
      if ( *(_BYTE *)v26 == 48 )
      {
        v32 = sub_B44E60(a1);
        sub_B448B0(v26, v32);
      }
      sub_BD84D0(a1, v31);
      sub_B43D60((_QWORD *)a1);
      nullsub_61();
      v101 = &unk_49DA100;
      nullsub_63();
      if ( v93 != (unsigned int *)&v95 )
        _libc_free((unsigned __int64)v93);
      v6 = v71;
      goto LABEL_9;
    }
LABEL_71:
    v92 = 257;
    v44 = sub_BD2C40(72, unk_3F10A14);
    v31 = (__int64)v44;
    if ( v44 )
      sub_B515B0((__int64)v44, v26, (__int64)v28, (__int64)&v89, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v100 + 16LL))(
      v100,
      v31,
      &v85,
      v97,
      v98);
    v45 = v93;
    v46 = &v93[4 * v94];
    if ( v93 != v46 )
    {
      do
      {
        v47 = *((_QWORD *)v45 + 1);
        v48 = *v45;
        v45 += 4;
        sub_B99FD0(v31, v48, v47);
      }
      while ( v46 != v45 );
    }
    goto LABEL_54;
  }
LABEL_9:
  if ( v84 > 0x40 && v83 )
    j_j___libc_free_0_0(v83);
  if ( v82 > 0x40 && v81 )
    j_j___libc_free_0_0(v81);
  if ( v80 > 0x40 && v79 )
    j_j___libc_free_0_0(v79);
  if ( v78 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  return v6;
}
