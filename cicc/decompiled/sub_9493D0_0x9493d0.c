// Function: sub_9493D0
// Address: 0x9493d0
//
__int64 __fastcall sub_9493D0(__int64 a1, int a2, int a3, __int64 a4, __m128i **a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r14
  __m128i *v12; // rax
  __m128i *v13; // r13
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, __int64, __int64); // rax
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned int *v18; // r12
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __m128i *v22; // rax
  _BYTE *v23; // r10
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, __int64, __int64); // rax
  unsigned int *v26; // r14
  unsigned int *v27; // r12
  unsigned int *v28; // rbx
  unsigned int *v29; // r14
  _BYTE *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rax
  int v34; // r14d
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // r14
  unsigned int *v39; // r14
  unsigned int *v40; // rbx
  __int64 v41; // rdx
  __int64 v42; // rsi
  __m128i *v43; // rax
  __m128i *v44; // r14
  __int64 v45; // rdi
  __int64 (__fastcall *v46)(__int64, __int64, __int64); // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned int *v49; // r12
  unsigned int *v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 result; // rax
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rax
  __int64 v59; // rbx
  __int64 v60; // rax
  unsigned __int64 v61; // rcx
  unsigned int *v62; // rax
  unsigned int *v63; // rcx
  unsigned int *v64; // rbx
  unsigned int *v65; // r12
  __int64 v66; // rdx
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rdx
  unsigned int *v70; // r12
  unsigned int *v71; // rbx
  __int64 v72; // rdx
  __int64 v73; // rsi
  unsigned int *v74; // r14
  unsigned int *v75; // r12
  unsigned int *v76; // rbx
  unsigned int *v77; // r14
  _BYTE *v78; // r12
  __int64 v79; // rdx
  __int64 v80; // rsi
  unsigned __int64 v81; // rcx
  __m128i *v82; // rax
  _BYTE *v83; // r10
  __int64 v84; // rdi
  __int64 (__fastcall *v85)(__int64, __int64, __int64); // rax
  __int64 v86; // rax
  __int64 v87; // r14
  unsigned int *v88; // r14
  _BYTE *v89; // r12
  unsigned int *v90; // rbx
  __int64 v91; // rdx
  __int64 v92; // rsi
  __int64 v93; // rax
  int v94; // r14d
  unsigned int *v95; // rax
  unsigned int *v96; // rcx
  unsigned int *v97; // rbx
  unsigned int *v98; // r14
  __int64 v99; // rdx
  __int64 v100; // rsi
  unsigned int *v101; // rax
  unsigned int *v102; // rcx
  unsigned int *v103; // rbx
  unsigned int *v104; // r14
  __int64 v105; // rdx
  __int64 v106; // rsi
  __int64 v107; // [rsp+10h] [rbp-100h]
  __int64 v108; // [rsp+10h] [rbp-100h]
  int v109; // [rsp+10h] [rbp-100h]
  __int64 v110; // [rsp+10h] [rbp-100h]
  __int64 v111; // [rsp+18h] [rbp-F8h]
  _BYTE *v112; // [rsp+18h] [rbp-F8h]
  __int64 v113; // [rsp+18h] [rbp-F8h]
  int v114; // [rsp+18h] [rbp-F8h]
  __int64 v115; // [rsp+18h] [rbp-F8h]
  __int64 v116; // [rsp+18h] [rbp-F8h]
  __int64 v117; // [rsp+18h] [rbp-F8h]
  __int64 v118; // [rsp+18h] [rbp-F8h]
  __int64 v119; // [rsp+18h] [rbp-F8h]
  _BYTE *v120; // [rsp+18h] [rbp-F8h]
  __int64 v121; // [rsp+18h] [rbp-F8h]
  __int64 v122; // [rsp+18h] [rbp-F8h]
  __int64 v124; // [rsp+28h] [rbp-E8h]
  __int64 v127; // [rsp+38h] [rbp-D8h]
  __int64 v128; // [rsp+40h] [rbp-D0h]
  __int64 v129; // [rsp+40h] [rbp-D0h]
  __int64 v130; // [rsp+40h] [rbp-D0h]
  __int64 v131; // [rsp+48h] [rbp-C8h]
  __int64 v132; // [rsp+48h] [rbp-C8h]
  __int64 v133; // [rsp+48h] [rbp-C8h]
  _BYTE v134[32]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v135; // [rsp+70h] [rbp-A0h]
  _BYTE v136[32]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v137; // [rsp+A0h] [rbp-70h]
  _BYTE v138[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v139; // [rsp+D0h] [rbp-40h]

  v7 = a1;
  v8 = *(_QWORD *)(a4 + 16);
  v9 = *(_QWORD *)(a1 + 40);
  v10 = v8;
  if ( a2 == 14 )
    v10 = *(_QWORD *)(v8 + 16);
  v11 = *(_QWORD *)(v10 + 16);
  v107 = v10;
  v124 = *(_QWORD *)(v11 + 16);
  v127 = sub_BCB2F0(v9);
  v128 = v7 + 48;
  v131 = sub_BCE760(v127, 0);
  v137 = 257;
  v12 = sub_92F410(v7, v8);
  v13 = v12;
  if ( v131 != v12->m128i_i64[1] )
  {
    if ( v12->m128i_i8[0] > 0x15u )
    {
      v139 = 257;
      v13 = (__m128i *)sub_B52210(v12, v131, v138, 0, 0);
      (*(void (__fastcall **)(_QWORD, __m128i *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
        *(_QWORD *)(v7 + 136),
        v13,
        v136,
        *(_QWORD *)(v7 + 104),
        *(_QWORD *)(v7 + 112));
      v62 = *(unsigned int **)(v7 + 48);
      v63 = &v62[4 * *(unsigned int *)(v7 + 56)];
      if ( v62 != v63 )
      {
        v117 = v7;
        v64 = *(unsigned int **)(v7 + 48);
        v65 = v63;
        do
        {
          v66 = *((_QWORD *)v64 + 1);
          v67 = *v64;
          v64 += 4;
          sub_B99FD0(v13, v67, v66);
        }
        while ( v65 != v64 );
        v7 = v117;
      }
    }
    else
    {
      v14 = *(_QWORD *)(v7 + 128);
      v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v14 + 136LL);
      if ( v15 == sub_928970 )
        v13 = (__m128i *)sub_ADAFB0(v13, v131);
      else
        v13 = (__m128i *)v15(v14, (__int64)v13, v131);
      if ( v13->m128i_i8[0] > 0x1Cu )
      {
        (*(void (__fastcall **)(_QWORD, __m128i *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
          *(_QWORD *)(v7 + 136),
          v13,
          v136,
          *(_QWORD *)(v7 + 104),
          *(_QWORD *)(v7 + 112));
        v16 = *(_QWORD *)(v7 + 48);
        v17 = 16LL * *(unsigned int *)(v7 + 56);
        if ( v16 != v16 + v17 )
        {
          v111 = v7;
          v18 = (unsigned int *)(v16 + v17);
          v19 = *(unsigned int **)(v7 + 48);
          do
          {
            v20 = *((_QWORD *)v19 + 1);
            v21 = *v19;
            v19 += 4;
            sub_B99FD0(v13, v21, v20);
          }
          while ( v18 != v19 );
          v7 = v111;
        }
      }
    }
  }
  v135 = 257;
  v137 = 257;
  v22 = sub_92F410(v7, v11);
  v23 = v22;
  if ( v131 != v22->m128i_i64[1] )
  {
    if ( v22->m128i_i8[0] > 0x15u )
    {
      v139 = 257;
      v118 = sub_B52210(v22, v131, v138, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
        *(_QWORD *)(v7 + 136),
        v118,
        v136,
        *(_QWORD *)(v128 + 56),
        *(_QWORD *)(v128 + 64));
      v74 = *(unsigned int **)(v7 + 48);
      v23 = (_BYTE *)v118;
      v75 = &v74[4 * *(unsigned int *)(v7 + 56)];
      if ( v74 != v75 )
      {
        v119 = v7;
        v76 = *(unsigned int **)(v7 + 48);
        v77 = v75;
        v78 = v23;
        do
        {
          v79 = *((_QWORD *)v76 + 1);
          v80 = *v76;
          v76 += 4;
          sub_B99FD0(v78, v80, v79);
        }
        while ( v77 != v76 );
        v7 = v119;
        LODWORD(v23) = (_DWORD)v78;
      }
    }
    else
    {
      v24 = *(_QWORD *)(v7 + 128);
      v25 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v24 + 136LL);
      if ( v25 == sub_928970 )
        v23 = (_BYTE *)sub_ADAFB0(v23, v131);
      else
        v23 = (_BYTE *)v25(v24, (__int64)v23, v131);
      if ( *v23 > 0x1Cu )
      {
        v112 = v23;
        (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
          *(_QWORD *)(v7 + 136),
          v23,
          v136,
          *(_QWORD *)(v128 + 56),
          *(_QWORD *)(v128 + 64));
        v26 = *(unsigned int **)(v7 + 48);
        v23 = v112;
        v27 = &v26[4 * *(unsigned int *)(v7 + 56)];
        if ( v26 != v27 )
        {
          v113 = v7;
          v28 = *(unsigned int **)(v7 + 48);
          v29 = v27;
          v30 = v23;
          do
          {
            v31 = *((_QWORD *)v28 + 1);
            v32 = *v28;
            v28 += 4;
            sub_B99FD0(v30, v32, v31);
          }
          while ( v29 != v28 );
          v7 = v113;
          LODWORD(v23) = (_DWORD)v30;
        }
      }
    }
  }
  v114 = (int)v23;
  v33 = sub_AA4E30(*(_QWORD *)(v7 + 96));
  v34 = (unsigned __int8)sub_AE5020(v33, v127);
  v139 = 257;
  v35 = sub_BD2C40(80, unk_3F10A14);
  v36 = v35;
  if ( v35 )
    sub_B4D190(v35, v127, v114, (unsigned int)v138, 0, v34, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
    *(_QWORD *)(v7 + 136),
    v36,
    v134,
    *(_QWORD *)(v128 + 56),
    *(_QWORD *)(v128 + 64));
  v37 = *(_QWORD *)(v7 + 48);
  v38 = 16LL * *(unsigned int *)(v7 + 56);
  if ( v37 != v37 + v38 )
  {
    v115 = v7;
    v39 = (unsigned int *)(v37 + v38);
    v40 = *(unsigned int **)(v7 + 48);
    do
    {
      v41 = *((_QWORD *)v40 + 1);
      v42 = *v40;
      v40 += 4;
      sub_B99FD0(v36, v42, v41);
    }
    while ( v39 != v40 );
    v7 = v115;
  }
  if ( a2 != 14 )
  {
    v116 = v36;
    goto LABEL_29;
  }
  v135 = 257;
  v137 = 257;
  v82 = sub_92F410(v7, v107);
  v83 = v82;
  if ( v131 != v82->m128i_i64[1] )
  {
    if ( v82->m128i_i8[0] > 0x15u )
    {
      v139 = 257;
      v122 = sub_B52210(v82, v131, v138, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
        *(_QWORD *)(v7 + 136),
        v122,
        v136,
        *(_QWORD *)(v128 + 56),
        *(_QWORD *)(v128 + 64));
      v101 = *(unsigned int **)(v7 + 48);
      v83 = (_BYTE *)v122;
      v102 = &v101[4 * *(unsigned int *)(v7 + 56)];
      if ( v101 == v102 )
        goto LABEL_72;
      v121 = v36;
      v89 = v83;
      v108 = v7;
      v103 = *(unsigned int **)(v7 + 48);
      v104 = v102;
      do
      {
        v105 = *((_QWORD *)v103 + 1);
        v106 = *v103;
        v103 += 4;
        sub_B99FD0(v89, v106, v105);
      }
      while ( v104 != v103 );
    }
    else
    {
      v84 = *(_QWORD *)(v7 + 128);
      v85 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v84 + 136LL);
      if ( v85 == sub_928970 )
        v83 = (_BYTE *)sub_ADAFB0(v83, v131);
      else
        v83 = (_BYTE *)v85(v84, (__int64)v83, v131);
      if ( *v83 <= 0x1Cu )
        goto LABEL_72;
      v120 = v83;
      (*(void (__fastcall **)(_QWORD, _BYTE *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
        *(_QWORD *)(v7 + 136),
        v83,
        v136,
        *(_QWORD *)(v128 + 56),
        *(_QWORD *)(v128 + 64));
      v86 = *(_QWORD *)(v7 + 48);
      v83 = v120;
      v87 = 16LL * *(unsigned int *)(v7 + 56);
      if ( v86 == v86 + v87 )
        goto LABEL_72;
      v121 = v36;
      v88 = (unsigned int *)(v86 + v87);
      v89 = v83;
      v108 = v7;
      v90 = *(unsigned int **)(v7 + 48);
      do
      {
        v91 = *((_QWORD *)v90 + 1);
        v92 = *v90;
        v90 += 4;
        sub_B99FD0(v89, v92, v91);
      }
      while ( v88 != v90 );
    }
    LODWORD(v83) = (_DWORD)v89;
    v7 = v108;
    v36 = v121;
  }
LABEL_72:
  v109 = (int)v83;
  v93 = sub_AA4E30(*(_QWORD *)(v7 + 96));
  v94 = (unsigned __int8)sub_AE5020(v93, v127);
  v139 = 257;
  v116 = sub_BD2C40(80, unk_3F10A14);
  if ( v116 )
    sub_B4D190(v116, v127, v109, (unsigned int)v138, 0, v94, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
    *(_QWORD *)(v7 + 136),
    v116,
    v134,
    *(_QWORD *)(v128 + 56),
    *(_QWORD *)(v128 + 64));
  v95 = *(unsigned int **)(v7 + 48);
  v96 = &v95[4 * *(unsigned int *)(v7 + 56)];
  if ( v95 != v96 )
  {
    v110 = v7;
    v97 = *(unsigned int **)(v7 + 48);
    v98 = v96;
    do
    {
      v99 = *((_QWORD *)v97 + 1);
      v100 = *v97;
      v97 += 4;
      sub_B99FD0(v116, v100, v99);
    }
    while ( v98 != v97 );
    v7 = v110;
  }
LABEL_29:
  v137 = 257;
  v43 = sub_92F410(v7, v124);
  v44 = v43;
  if ( v131 != v43->m128i_i64[1] )
  {
    if ( v43->m128i_i8[0] > 0x15u )
    {
      v139 = 257;
      v44 = (__m128i *)sub_B52210(v43, v131, v138, 0, 0);
      (*(void (__fastcall **)(_QWORD, __m128i *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
        *(_QWORD *)(v7 + 136),
        v44,
        v136,
        *(_QWORD *)(v128 + 56),
        *(_QWORD *)(v128 + 64));
      v68 = *(_QWORD *)(v7 + 48);
      v69 = 16LL * *(unsigned int *)(v7 + 56);
      if ( v68 != v68 + v69 )
      {
        v133 = v36;
        v70 = (unsigned int *)(v68 + v69);
        v130 = v7;
        v71 = *(unsigned int **)(v7 + 48);
        do
        {
          v72 = *((_QWORD *)v71 + 1);
          v73 = *v71;
          v71 += 4;
          sub_B99FD0(v44, v73, v72);
        }
        while ( v70 != v71 );
        v36 = v133;
        v7 = v130;
      }
    }
    else
    {
      v45 = *(_QWORD *)(v7 + 128);
      v46 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v45 + 136LL);
      if ( v46 == sub_928970 )
        v44 = (__m128i *)sub_ADAFB0(v44, v131);
      else
        v44 = (__m128i *)v46(v45, (__int64)v44, v131);
      if ( v44->m128i_i8[0] > 0x1Cu )
      {
        (*(void (__fastcall **)(_QWORD, __m128i *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 136) + 16LL))(
          *(_QWORD *)(v7 + 136),
          v44,
          v136,
          *(_QWORD *)(v128 + 56),
          *(_QWORD *)(v128 + 64));
        v47 = *(_QWORD *)(v7 + 48);
        v48 = 16LL * *(unsigned int *)(v7 + 56);
        if ( v47 != v47 + v48 )
        {
          v132 = v36;
          v49 = (unsigned int *)(v47 + v48);
          v129 = v7;
          v50 = *(unsigned int **)(v7 + 48);
          do
          {
            v51 = *((_QWORD *)v50 + 1);
            v52 = *v50;
            v50 += 4;
            sub_B99FD0(v44, v52, v51);
          }
          while ( v49 != v50 );
          v36 = v132;
          v7 = v129;
        }
      }
    }
  }
  *a5 = v44;
  if ( !a3 )
  {
    v53 = *(unsigned int *)(a6 + 8);
    v54 = v53 + 1;
    if ( v53 + 1 <= (unsigned __int64)*(unsigned int *)(a6 + 12) )
      goto LABEL_40;
    goto LABEL_46;
  }
  v57 = (unsigned __int8)a2 << 16;
  LOBYTE(v57) = (16 * a3) | 5;
  v58 = sub_BCB2D0(*(_QWORD *)(v7 + 40));
  v59 = sub_ACD640(v58, v57, 0);
  v60 = *(unsigned int *)(a6 + 8);
  if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, a6 + 16, v60 + 1, 8);
    v60 = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v60) = v59;
  v61 = *(unsigned int *)(a6 + 12);
  v53 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  v54 = v53 + 1;
  *(_DWORD *)(a6 + 8) = v53;
  if ( v53 + 1 > v61 )
  {
LABEL_46:
    sub_C8D5F0(a6, a6 + 16, v54, 8);
    v53 = *(unsigned int *)(a6 + 8);
  }
LABEL_40:
  *(_QWORD *)(*(_QWORD *)a6 + 8 * v53) = v13;
  result = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  *(_DWORD *)(a6 + 8) = result;
  if ( a2 != 14 )
  {
    v56 = result + 1;
    if ( result + 1 <= (unsigned __int64)*(unsigned int *)(a6 + 12) )
      goto LABEL_42;
LABEL_62:
    sub_C8D5F0(a6, a6 + 16, v56, 8);
    result = *(unsigned int *)(a6 + 8);
    goto LABEL_42;
  }
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
  {
    sub_C8D5F0(a6, a6 + 16, result + 1, 8);
    result = *(unsigned int *)(a6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = v116;
  v81 = *(unsigned int *)(a6 + 12);
  result = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
  v56 = result + 1;
  *(_DWORD *)(a6 + 8) = result;
  if ( result + 1 > v81 )
    goto LABEL_62;
LABEL_42:
  *(_QWORD *)(*(_QWORD *)a6 + 8 * result) = v36;
  ++*(_DWORD *)(a6 + 8);
  return result;
}
