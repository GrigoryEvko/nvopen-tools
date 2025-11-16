// Function: sub_2632600
// Address: 0x2632600
//
__int64 __fastcall sub_2632600(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 *v7; // r15
  __int64 v8; // r13
  __int64 *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // rbx
  __int16 v18; // ax
  __int64 v19; // rax
  char v20; // si^1
  char v21; // si
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // r12
  __int64 v25; // rsi
  unsigned int *v26; // r14
  unsigned int *v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 *v31; // rbx
  unsigned int v32; // r12d
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // r12d
  __int64 i; // r15
  __int64 v37; // r13
  __int64 v38; // rax
  unsigned __int64 v39; // rbx
  int v40; // eax
  unsigned __int64 v41; // rbx
  bool v42; // cf
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int8 *v45; // r12
  __int64 (__fastcall *v46)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v47; // rsi
  __int64 v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // r9
  __int64 v55; // r8
  __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // rcx
  _QWORD **v62; // rdx
  int v63; // ecx
  __int64 *v64; // rax
  __int64 v65; // rsi
  unsigned int *v66; // r13
  unsigned int *v67; // r12
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 *v71; // r9
  __int64 *v72; // rax
  unsigned __int64 v73; // r14
  __int64 v74; // rax
  __int64 v75; // r12
  __int64 *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r14
  __int64 v79; // r12
  _QWORD *v80; // rdi
  const char *v81; // r12
  __int64 v82; // r14
  size_t v83; // rax
  __int64 v84; // [rsp-8h] [rbp-208h]
  char v85; // [rsp+10h] [rbp-1F0h]
  __int64 v86; // [rsp+10h] [rbp-1F0h]
  __int64 v89; // [rsp+20h] [rbp-1E0h]
  __int64 *v90; // [rsp+28h] [rbp-1D8h]
  unsigned int v91; // [rsp+40h] [rbp-1C0h]
  __int64 v93; // [rsp+50h] [rbp-1B0h]
  __int64 v94; // [rsp+50h] [rbp-1B0h]
  __int64 v95; // [rsp+58h] [rbp-1A8h]
  char v96; // [rsp+59h] [rbp-1A7h]
  __int64 v97; // [rsp+68h] [rbp-198h]
  char v98[32]; // [rsp+70h] [rbp-190h] BYREF
  __int16 v99; // [rsp+90h] [rbp-170h]
  _BYTE v100[32]; // [rsp+A0h] [rbp-160h] BYREF
  __int16 v101; // [rsp+C0h] [rbp-140h]
  __int64 v102; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v103; // [rsp+D8h] [rbp-128h]
  __int64 v104; // [rsp+E0h] [rbp-120h]
  __int64 v105; // [rsp+E8h] [rbp-118h]
  __int64 *v106; // [rsp+F0h] [rbp-110h]
  __int64 v107; // [rsp+F8h] [rbp-108h]
  _BYTE v108[64]; // [rsp+100h] [rbp-100h] BYREF
  unsigned int *v109; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v110; // [rsp+148h] [rbp-B8h]
  _BYTE v111[16]; // [rsp+150h] [rbp-B0h] BYREF
  __int16 v112; // [rsp+160h] [rbp-A0h]
  __int64 v113; // [rsp+170h] [rbp-90h]
  __int64 v114; // [rsp+178h] [rbp-88h]
  __int64 v115; // [rsp+180h] [rbp-80h]
  __int64 v116; // [rsp+188h] [rbp-78h]
  void **v117; // [rsp+190h] [rbp-70h]
  _QWORD *v118; // [rsp+198h] [rbp-68h]
  __int64 v119; // [rsp+1A0h] [rbp-60h]
  int v120; // [rsp+1A8h] [rbp-58h]
  __int16 v121; // [rsp+1ACh] [rbp-54h]
  char v122; // [rsp+1AEh] [rbp-52h]
  __int64 v123; // [rsp+1B0h] [rbp-50h]
  __int64 v124; // [rsp+1B8h] [rbp-48h]
  void *v125; // [rsp+1C0h] [rbp-40h] BYREF
  _QWORD v126[7]; // [rsp+1C8h] [rbp-38h] BYREF

  v6 = a2;
  v106 = (__int64 *)v108;
  v107 = 0x800000000LL;
  v85 = a4;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  sub_2631930(a2, (__int64)&v102, a3, a4, a5, a6);
  v90 = &v106[(unsigned int)v107];
  if ( v106 != v90 )
  {
    v7 = v106;
    do
    {
      v8 = *v7;
      if ( a1[24] != (__int64 *)*v7 )
      {
        v9 = a1[23];
        if ( !v9 )
        {
          v112 = 259;
          v71 = *a1;
          v109 = (unsigned int *)"__cfi_global_var_init";
          v93 = (__int64)v71;
          v91 = *((_DWORD *)v71 + 80);
          v72 = (__int64 *)sub_BCB120((_QWORD *)*v71);
          v73 = sub_BCF640(v72, 0);
          v74 = sub_BD2DA0(136);
          v75 = v74;
          if ( v74 )
            sub_B2C3B0(v74, v73, 7, v91, (__int64)&v109, v93);
          HIBYTE(v112) = 1;
          v109 = (unsigned int *)"entry";
          a1[23] = (__int64 *)v75;
          v76 = *a1;
          LOBYTE(v112) = 3;
          v94 = *v76;
          v77 = sub_22077B0(0x50u);
          v78 = v77;
          if ( v77 )
            sub_AA4D50(v77, v94, (__int64)&v109, v75, 0);
          v79 = **a1;
          sub_B43C20((__int64)&v109, v78);
          v80 = sub_BD2C40(72, 0);
          if ( v80 )
            sub_B4BB80((__int64)v80, v79, 0, 0, (__int64)v109, v110);
          v81 = ".text.startup";
          v82 = (__int64)a1[23];
          if ( *((_DWORD *)a1 + 9) == 5 )
            v81 = "__TEXT,__StaticInit,regular,pure_instructions";
          v83 = strlen(v81);
          sub_B31A00(v82, (__int64)v81, v83);
          sub_2A3ED40(*a1, a1[23], 0, 0);
          v9 = a1[23];
        }
        v10 = v9[10];
        if ( !v10 )
          BUG();
        v11 = *(_QWORD *)(v10 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v11 == v10 + 24 )
        {
          v13 = 0;
        }
        else
        {
          if ( !v11 )
            BUG();
          v12 = *(unsigned __int8 *)(v11 - 24);
          v13 = 0;
          v14 = v11 - 24;
          if ( (unsigned int)(v12 - 30) < 0xB )
            v13 = v14;
        }
        v15 = sub_BD5C60(v13);
        v122 = 7;
        v116 = v15;
        v117 = &v125;
        v118 = v126;
        v125 = &unk_49DA100;
        v109 = (unsigned int *)v111;
        v110 = 0x200000000LL;
        v126[0] = &unk_49DA0B0;
        v119 = 0;
        v120 = 0;
        v121 = 512;
        v123 = 0;
        v124 = 0;
        v113 = 0;
        v114 = 0;
        LOWORD(v115) = 0;
        sub_D5F1F0((__int64)&v109, v13);
        v16 = *(_WORD *)(v8 + 34);
        *(_BYTE *)(v8 + 80) &= ~1u;
        v17 = *(_QWORD *)(v8 - 32);
        v18 = (v16 >> 1) & 0x3F;
        if ( v18 )
        {
          v20 = v96;
          v21 = v18 - 1;
        }
        else
        {
          v19 = sub_AA4E30(v113);
          v20 = v96;
          v21 = sub_AE5020(v19, *(_QWORD *)(v17 + 8));
        }
        v96 = v20;
        v101 = 257;
        v22 = sub_BD2C40(80, unk_3F10A10);
        v24 = (__int64)v22;
        if ( v22 )
        {
          sub_B4D3C0((__int64)v22, v17, v8, 0, v21, v23, 0, 0);
          v23 = v84;
        }
        v25 = v24;
        (*(void (__fastcall **)(_QWORD *, __int64, _BYTE *, __int64, __int64, __int64))(*v118 + 16LL))(
          v118,
          v24,
          v100,
          v114,
          v115,
          v23);
        v26 = v109;
        v27 = &v109[4 * (unsigned int)v110];
        if ( v109 != v27 )
        {
          do
          {
            v28 = *((_QWORD *)v26 + 1);
            v25 = *v26;
            v26 += 4;
            sub_B99FD0(v24, v25, v28);
          }
          while ( v27 != v26 );
        }
        v29 = sub_AD6530(*(_QWORD *)(v8 + 24), v25);
        sub_B30160(v8, v29);
        nullsub_61();
        v125 = &unk_49DA100;
        nullsub_63();
        if ( v109 != (unsigned int *)v111 )
          _libc_free((unsigned __int64)v109);
      }
      ++v7;
    }
    while ( v90 != v7 );
    v6 = a2;
  }
  v30 = *(_QWORD *)(v6 + 24);
  v112 = 257;
  v31 = *a1;
  v32 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 8LL);
  v89 = sub_BD2DA0(136);
  v35 = v32 >> 8;
  if ( v89 )
    sub_B2C3B0(v89, v30, 9, v35, (__int64)&v109, (__int64)v31);
  sub_2631F50((__int64)a1, v6, v89, v85, v33, v34);
  v109 = (unsigned int *)v89;
  sub_3144670(&v109, 1, 0, 1, 0);
  for ( i = *(_QWORD *)(v89 + 16); i; i = *(_QWORD *)(v89 + 16) )
  {
    v37 = *(_QWORD *)(i + 24);
    if ( *(_BYTE *)v37 <= 0x1Cu )
      BUG();
    if ( *(_BYTE *)v37 == 84 )
    {
      v38 = *(_QWORD *)(*(_QWORD *)(v37 - 8)
                      + 32LL * *(unsigned int *)(v37 + 72)
                      + 8LL * (unsigned int)((i - *(_QWORD *)(v37 - 8)) >> 5));
      v39 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v39 == v38 + 48 )
      {
        v95 = 0;
      }
      else
      {
        if ( !v39 )
          BUG();
        v40 = *(unsigned __int8 *)(v39 - 24);
        v41 = v39 - 24;
        v42 = (unsigned int)(v40 - 30) < 0xB;
        v43 = 0;
        if ( v42 )
          v43 = v41;
        v95 = v43;
      }
    }
    else
    {
      v95 = *(_QWORD *)(i + 24);
      v37 = 0;
    }
    v44 = sub_BD5C60(v95);
    v122 = 7;
    v116 = v44;
    v117 = &v125;
    v118 = v126;
    v110 = 0x200000000LL;
    v125 = &unk_49DA100;
    LOWORD(v115) = 0;
    v121 = 512;
    v126[0] = &unk_49DA0B0;
    v109 = (unsigned int *)v111;
    v119 = 0;
    v120 = 0;
    v123 = 0;
    v124 = 0;
    v113 = 0;
    v114 = 0;
    sub_D5F1F0((__int64)&v109, v95);
    v99 = 257;
    v45 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v6 + 8), v95);
    v46 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v117 + 7);
    if ( v46 == sub_928890 )
    {
      if ( *(_BYTE *)v6 > 0x15u || *v45 > 0x15u )
      {
LABEL_53:
        v101 = 257;
        v48 = (__int64)sub_BD2C40(72, unk_3F10FD0);
        if ( v48 )
        {
          v62 = *(_QWORD ***)(v6 + 8);
          v63 = *((unsigned __int8 *)v62 + 8);
          if ( (unsigned int)(v63 - 17) > 1 )
          {
            v65 = sub_BCB2A0(*v62);
          }
          else
          {
            BYTE4(v97) = (_BYTE)v63 == 18;
            LODWORD(v97) = *((_DWORD *)v62 + 8);
            v64 = (__int64 *)sub_BCB2A0(*v62);
            v65 = sub_BCE1B0(v64, v97);
          }
          sub_B523C0(v48, v65, 53, 33, v6, (__int64)v45, (__int64)v100, 0, 0, 0);
        }
        v47 = v48;
        (*(void (__fastcall **)(_QWORD *, __int64, char *, __int64, __int64))(*v118 + 16LL))(v118, v48, v98, v114, v115);
        if ( v109 != &v109[4 * (unsigned int)v110] )
        {
          v86 = v37;
          v66 = v109;
          v67 = &v109[4 * (unsigned int)v110];
          do
          {
            v68 = *((_QWORD *)v66 + 1);
            v47 = *v66;
            v66 += 4;
            sub_B99FD0(v48, v47, v68);
          }
          while ( v67 != v66 );
          v37 = v86;
        }
        goto LABEL_35;
      }
      v47 = v6;
      v48 = sub_AAB310(0x21u, (unsigned __int8 *)v6, v45);
    }
    else
    {
      v47 = 33;
      v48 = v46((__int64)v117, 33u, (_BYTE *)v6, v45);
    }
    if ( !v48 )
      goto LABEL_53;
LABEL_35:
    v49 = *(_QWORD *)(v6 + 8);
    v101 = 257;
    v50 = sub_AD6530(v49, v47);
    v51 = sub_B36550(&v109, v48, a3, v50, (__int64)v100, 0);
    v52 = v51;
    if ( v37 )
    {
      v53 = 0;
      v54 = v51 + 16;
      v55 = *(_QWORD *)(v95 + 40);
      v56 = 8LL * (*(_DWORD *)(v37 + 4) & 0x7FFFFFF);
      if ( (*(_DWORD *)(v37 + 4) & 0x7FFFFFF) != 0 )
      {
        do
        {
          while ( 1 )
          {
            v57 = *(_QWORD *)(v37 - 8);
            if ( v55 == *(_QWORD *)(v57 + 32LL * *(unsigned int *)(v37 + 72) + v53) )
            {
              v58 = v57 + 4 * v53;
              if ( *(_QWORD *)v58 )
              {
                v59 = *(_QWORD *)(v58 + 8);
                **(_QWORD **)(v58 + 16) = v59;
                if ( v59 )
                  *(_QWORD *)(v59 + 16) = *(_QWORD *)(v58 + 16);
              }
              *(_QWORD *)v58 = v52;
              if ( v52 )
                break;
            }
            v53 += 8;
            if ( v53 == v56 )
              goto LABEL_47;
          }
          v60 = *(_QWORD *)(v52 + 16);
          *(_QWORD *)(v58 + 8) = v60;
          if ( v60 )
            *(_QWORD *)(v60 + 16) = v58 + 8;
          v53 += 8;
          *(_QWORD *)(v58 + 16) = v54;
          *(_QWORD *)(v52 + 16) = v58;
        }
        while ( v53 != v56 );
      }
    }
    else
    {
      if ( *(_QWORD *)i )
      {
        v69 = *(_QWORD *)(i + 8);
        **(_QWORD **)(i + 16) = v69;
        if ( v69 )
          *(_QWORD *)(v69 + 16) = *(_QWORD *)(i + 16);
      }
      *(_QWORD *)i = v52;
      if ( v52 )
      {
        v70 = *(_QWORD *)(v52 + 16);
        *(_QWORD *)(i + 8) = v70;
        if ( v70 )
          *(_QWORD *)(v70 + 16) = i + 8;
        *(_QWORD *)(i + 16) = v52 + 16;
        *(_QWORD *)(v52 + 16) = i;
      }
    }
LABEL_47:
    nullsub_61();
    v125 = &unk_49DA100;
    nullsub_63();
    if ( v109 != (unsigned int *)v111 )
      _libc_free((unsigned __int64)v109);
  }
  sub_B2E860((_QWORD *)v89);
  if ( v106 != (__int64 *)v108 )
    _libc_free((unsigned __int64)v106);
  return sub_C7D6A0(v103, 8LL * (unsigned int)v105, 8);
}
