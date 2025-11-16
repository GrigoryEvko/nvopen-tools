// Function: sub_300FDC0
// Address: 0x300fdc0
//
void __fastcall sub_300FDC0(_QWORD *a1, __int64 a2, char a3, unsigned int a4)
{
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r15
  unsigned int *v13; // rax
  int v14; // ecx
  unsigned int *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  _QWORD *v19; // r14
  _QWORD *v20; // rsi
  _BYTE *v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  unsigned int *v33; // rax
  int v34; // ecx
  unsigned int *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // r9
  __int64 v45; // r14
  __int64 v46; // r13
  unsigned int *v47; // r13
  unsigned int *v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // rdx
  __int64 v52; // r10
  unsigned __int64 v53; // rsi
  __int64 v54; // r13
  __int64 v55; // rax
  char v56; // al
  char v57; // r15
  __int64 v58; // r9
  _QWORD *v59; // r14
  unsigned int *v60; // r15
  unsigned int *v61; // r13
  __int64 v62; // rdx
  unsigned int v63; // esi
  __m128i v64; // xmm0
  _QWORD *v65; // rax
  __int64 v66; // r15
  __int64 v67; // r14
  _QWORD *v68; // rax
  __int64 v69; // r13
  unsigned int *v70; // r15
  unsigned int *v71; // r14
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 *v74; // rax
  __int64 v75; // r15
  __int64 v76; // r14
  __int64 v77; // rax
  char v78; // bl
  _QWORD *v79; // rax
  __int64 v80; // r13
  unsigned int *v81; // r12
  unsigned int *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  int v85; // r14d
  __int64 *v86; // rax
  unsigned __int64 v87; // rsi
  unsigned __int64 v88; // rsi
  __int64 v89; // [rsp-10h] [rbp-220h]
  __int64 v90; // [rsp+0h] [rbp-210h]
  char v91; // [rsp+18h] [rbp-1F8h]
  _QWORD *v93; // [rsp+20h] [rbp-1F0h]
  __int64 v94; // [rsp+20h] [rbp-1F0h]
  __int64 v95; // [rsp+28h] [rbp-1E8h]
  __int64 *v96; // [rsp+28h] [rbp-1E8h]
  __int64 v98; // [rsp+30h] [rbp-1E0h]
  __int64 v99; // [rsp+30h] [rbp-1E0h]
  _QWORD *v100; // [rsp+58h] [rbp-1B8h]
  __int64 v101; // [rsp+88h] [rbp-188h] BYREF
  __m128i *v102; // [rsp+90h] [rbp-180h]
  __int64 v103; // [rsp+98h] [rbp-178h]
  __m128i v104; // [rsp+A0h] [rbp-170h] BYREF
  char v105[32]; // [rsp+B0h] [rbp-160h] BYREF
  __int16 v106; // [rsp+D0h] [rbp-140h]
  _QWORD v107[4]; // [rsp+E0h] [rbp-130h] BYREF
  __int16 v108; // [rsp+100h] [rbp-110h]
  unsigned __int64 v109[2]; // [rsp+110h] [rbp-100h] BYREF
  __m128i v110; // [rsp+120h] [rbp-F0h] BYREF
  _QWORD *v111; // [rsp+130h] [rbp-E0h]
  _QWORD *v112; // [rsp+138h] [rbp-D8h]
  _QWORD *v113; // [rsp+140h] [rbp-D0h]
  unsigned int *v114; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v115; // [rsp+158h] [rbp-B8h]
  _BYTE v116[32]; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+180h] [rbp-90h]
  __int64 v118; // [rsp+188h] [rbp-88h]
  __int64 v119; // [rsp+190h] [rbp-80h]
  _QWORD *v120; // [rsp+198h] [rbp-78h]
  void **v121; // [rsp+1A0h] [rbp-70h]
  void **v122; // [rsp+1A8h] [rbp-68h]
  __int64 v123; // [rsp+1B0h] [rbp-60h]
  int v124; // [rsp+1B8h] [rbp-58h]
  __int16 v125; // [rsp+1BCh] [rbp-54h]
  char v126; // [rsp+1BEh] [rbp-52h]
  __int64 v127; // [rsp+1C0h] [rbp-50h]
  __int64 v128; // [rsp+1C8h] [rbp-48h]
  void *v129; // [rsp+1D0h] [rbp-40h] BYREF
  void *v130; // [rsp+1D8h] [rbp-38h] BYREF

  v123 = 0;
  v120 = (_QWORD *)sub_AA48A0(a2);
  v121 = &v129;
  v122 = &v130;
  v114 = (unsigned int *)v116;
  v129 = &unk_49DA100;
  v115 = 0x200000000LL;
  v124 = 0;
  v125 = 512;
  v126 = 7;
  v127 = 0;
  v128 = 0;
  v117 = 0;
  v118 = 0;
  LOWORD(v119) = 0;
  v130 = &unk_49DA0B0;
  v6 = sub_AA5190(a2);
  v8 = v6;
  if ( !v6 )
  {
    v117 = a2;
    v118 = 0;
    LOWORD(v119) = 0;
    goto LABEL_3;
  }
  v118 = v6;
  v8 = v6 - 24;
  v117 = a2;
  LOWORD(v119) = v7;
  if ( v6 != a2 + 48 )
  {
LABEL_3:
    v9 = *(_QWORD *)sub_B46C60(v8);
    v109[0] = v9;
    if ( v9 && (sub_B96E90((__int64)v109, v9, 1), (v12 = v109[0]) != 0) )
    {
      v13 = v114;
      v14 = v115;
      v15 = &v114[4 * (unsigned int)v115];
      if ( v114 != v15 )
      {
        while ( *v13 )
        {
          v13 += 4;
          if ( v15 == v13 )
            goto LABEL_75;
        }
        *((_QWORD *)v13 + 1) = v109[0];
LABEL_10:
        sub_B91220((__int64)v109, v12);
        goto LABEL_11;
      }
LABEL_75:
      if ( (unsigned int)v115 >= (unsigned __int64)HIDWORD(v115) )
      {
        v87 = (unsigned int)v115 + 1LL;
        if ( HIDWORD(v115) < v87 )
        {
          sub_C8D5F0((__int64)&v114, v116, v87, 0x10u, v10, v11);
          v15 = &v114[4 * (unsigned int)v115];
        }
        *(_QWORD *)v15 = 0;
        *((_QWORD *)v15 + 1) = v12;
        v12 = v109[0];
        LODWORD(v115) = v115 + 1;
      }
      else
      {
        if ( v15 )
        {
          *v15 = 0;
          *((_QWORD *)v15 + 1) = v12;
          v14 = v115;
          v12 = v109[0];
        }
        LODWORD(v115) = v14 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v114, 0);
      v12 = v109[0];
    }
    if ( !v12 )
      goto LABEL_11;
    goto LABEL_10;
  }
LABEL_11:
  v16 = sub_AA4FF0(a2);
  v17 = v16;
  if ( !v16 )
    BUG();
  v18 = *(_QWORD *)(v16 - 8);
  v19 = 0;
  v20 = 0;
  if ( v18 )
  {
    do
    {
      while ( 1 )
      {
        v21 = *(_BYTE **)(v18 + 24);
        if ( *v21 == 85 )
          break;
        v18 = *(_QWORD *)(v18 + 8);
        if ( !v18 )
          goto LABEL_21;
      }
      v22 = *((_QWORD *)v21 - 4);
      v18 = *(_QWORD *)(v18 + 8);
      if ( a1[8] == v22 )
        v19 = v21;
      if ( a1[10] == v22 )
        v20 = v21;
    }
    while ( v18 );
LABEL_21:
    v100 = v20;
    if ( v19 )
    {
      v109[0] = (unsigned __int64)"exn";
      LOWORD(v111) = 259;
      v23 = sub_BCB2D0(v120);
      v24 = sub_ACD640(v23, 0, 0);
      v25 = a1[9];
      v26 = 0;
      v107[0] = v24;
      if ( v25 )
        v26 = *(_QWORD *)(v25 + 24);
      v95 = sub_921880(&v114, v26, v25, (int)v107, 1, (__int64)v109, 0);
      sub_BD84D0((__int64)v19, v95);
      sub_B43D60(v19);
      if ( !a3 )
      {
        if ( v100 )
          sub_B43D60(v100);
        goto LABEL_27;
      }
      v27 = *(_QWORD *)(v95 + 32);
      if ( v27 == *(_QWORD *)(v95 + 40) + 48LL || !v27 )
        BUG();
      v28 = *(_QWORD *)(v27 + 16);
      v118 = *(_QWORD *)(v95 + 32);
      LOWORD(v119) = 0;
      v117 = v28;
      v29 = *(_QWORD *)sub_B46C60(v27 - 24);
      v109[0] = v29;
      if ( v29 && (sub_B96E90((__int64)v109, v29, 1), (v32 = v109[0]) != 0) )
      {
        v33 = v114;
        v34 = v115;
        v35 = &v114[4 * (unsigned int)v115];
        if ( v114 != v35 )
        {
          while ( *v33 )
          {
            v33 += 4;
            if ( v35 == v33 )
              goto LABEL_83;
          }
          *((_QWORD *)v33 + 1) = v109[0];
LABEL_39:
          sub_B91220((__int64)v109, v32);
LABEL_40:
          LOWORD(v111) = 257;
          v98 = v17 - 24;
          v107[0] = v17 - 24;
          v36 = sub_BCB2D0(v120);
          v37 = sub_ACD640(v36, a4, 0);
          v38 = a1[6];
          v39 = 0;
          v107[1] = v37;
          if ( v38 )
            v39 = *(_QWORD *)(v38 + 24);
          sub_921880(&v114, v39, v38, (int)v107, 2, (__int64)v109, 0);
          v90 = a1[2];
          v40 = sub_BCB2D0(v120);
          v41 = sub_ACD640(v40, a4, 0);
          v42 = sub_AA4E30(v117);
          v91 = sub_AE5020(v42, *(_QWORD *)(v41 + 8));
          LOWORD(v111) = 257;
          v43 = sub_BD2C40(80, unk_3F10A10);
          v44 = v89;
          v45 = (__int64)v43;
          if ( v43 )
            sub_B4D3C0((__int64)v43, v41, v90, 0, v91, v90, 0, 0);
          (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64, __int64))*v122 + 2))(
            v122,
            v45,
            v109,
            v118,
            v119,
            v44);
          v46 = 4LL * (unsigned int)v115;
          if ( v114 != &v114[v46] )
          {
            v93 = a1;
            v47 = &v114[v46];
            v48 = v114;
            do
            {
              v49 = *((_QWORD *)v48 + 1);
              v50 = *v48;
              v48 += 4;
              sub_B99FD0(v45, v50, v49);
            }
            while ( v47 != v48 );
            a1 = v93;
          }
          v51 = a1[7];
          v52 = a1[3];
          v53 = 0;
          v108 = 257;
          if ( v51 )
            v53 = *(_QWORD *)(v51 + 24);
          v94 = v52;
          v54 = sub_921880(&v114, v53, v51, 0, 0, (__int64)v107, 0);
          v55 = sub_AA4E30(v117);
          v56 = sub_AE5020(v55, *(_QWORD *)(v54 + 8));
          LOWORD(v111) = 257;
          v57 = v56;
          v59 = sub_BD2C40(80, unk_3F10A10);
          if ( v59 )
            sub_B4D3C0((__int64)v59, v54, v94, 0, v57, v58, 0, 0);
          (*((void (__fastcall **)(void **, _QWORD *, unsigned __int64 *, __int64, __int64))*v122 + 2))(
            v122,
            v59,
            v109,
            v118,
            v119);
          v60 = v114;
          v61 = &v114[4 * (unsigned int)v115];
          if ( v114 != v61 )
          {
            do
            {
              v62 = *((_QWORD *)v60 + 1);
              v63 = *v60;
              v60 += 4;
              sub_B99FD0((__int64)v59, v63, v62);
            }
            while ( v61 != v60 );
          }
          v104.m128i_i64[0] = 0x74656C636E7566LL;
          v64 = _mm_load_si128(&v104);
          v102 = &v104;
          v106 = 257;
          v110 = v64;
          v109[0] = (unsigned __int64)&v110;
          v109[1] = 7;
          v103 = 0;
          v104.m128i_i8[0] = 0;
          v111 = 0;
          v112 = 0;
          v113 = 0;
          v65 = (_QWORD *)sub_22077B0(8u);
          v66 = a1[11];
          v67 = a1[12];
          v111 = v65;
          *v65 = v98;
          v113 = v65 + 1;
          v112 = v65 + 1;
          v101 = v95;
          v108 = 257;
          v68 = sub_BD2CC0(88, 0x1000000003uLL);
          v69 = (__int64)v68;
          if ( v68 )
          {
            v99 = (__int64)v68;
            sub_B44260((__int64)v68, **(_QWORD **)(v66 + 16), 56, 0x10000003u, 0, 0);
            *(_QWORD *)(v69 + 72) = 0;
            sub_B4A290(v69, v66, v67, &v101, 1, (__int64)v107, (__int64)v109, 1);
          }
          else
          {
            v99 = 0;
          }
          v96 = (__int64 *)(v69 + 72);
          if ( (_BYTE)v125 )
          {
            v86 = (__int64 *)sub_BD5C60(v99);
            *(_QWORD *)(v69 + 72) = sub_A7A090(v96, v86, -1, 72);
          }
          if ( (unsigned __int8)sub_920620(v99) )
          {
            v85 = v124;
            if ( v123 )
              sub_B99FD0(v69, 3u, v123);
            sub_B45150(v69, v85);
          }
          (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v122 + 2))(v122, v69, v105, v118, v119);
          v70 = v114;
          v71 = &v114[4 * (unsigned int)v115];
          if ( v114 != v71 )
          {
            do
            {
              v72 = *((_QWORD *)v70 + 1);
              v73 = *v70;
              v70 += 4;
              sub_B99FD0(v69, v73, v72);
            }
            while ( v71 != v70 );
          }
          if ( v111 )
            j_j___libc_free_0((unsigned __int64)v111);
          if ( (__m128i *)v109[0] != &v110 )
            j_j___libc_free_0(v109[0]);
          if ( v102 != &v104 )
            j_j___libc_free_0((unsigned __int64)v102);
          v74 = (__int64 *)sub_BD5C60(v99);
          *(_QWORD *)(v69 + 72) = sub_A7A090(v96, v74, -1, 41);
          v75 = a1[4];
          v76 = sub_BCB2D0(v120);
          v108 = 259;
          v107[0] = "selector";
          v77 = sub_AA4E30(v117);
          v78 = sub_AE5020(v77, v76);
          LOWORD(v111) = 257;
          v79 = sub_BD2C40(80, 1u);
          v80 = (__int64)v79;
          if ( v79 )
            sub_B4D190((__int64)v79, v76, v75, (__int64)v109, 0, v78, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v122 + 2))(
            v122,
            v80,
            v107,
            v118,
            v119);
          v81 = v114;
          v82 = &v114[4 * (unsigned int)v115];
          if ( v114 != v82 )
          {
            do
            {
              v83 = *((_QWORD *)v81 + 1);
              v84 = *v81;
              v81 += 4;
              sub_B99FD0(v80, v84, v83);
            }
            while ( v82 != v81 );
          }
          sub_BD84D0((__int64)v100, v80);
          sub_B43D60(v100);
          goto LABEL_27;
        }
LABEL_83:
        if ( (unsigned int)v115 >= (unsigned __int64)HIDWORD(v115) )
        {
          v88 = (unsigned int)v115 + 1LL;
          if ( HIDWORD(v115) < v88 )
          {
            sub_C8D5F0((__int64)&v114, v116, v88, 0x10u, v30, v31);
            v35 = &v114[4 * (unsigned int)v115];
          }
          *(_QWORD *)v35 = 0;
          *((_QWORD *)v35 + 1) = v32;
          v32 = v109[0];
          LODWORD(v115) = v115 + 1;
        }
        else
        {
          if ( v35 )
          {
            *v35 = 0;
            *((_QWORD *)v35 + 1) = v32;
            v34 = v115;
            v32 = v109[0];
          }
          LODWORD(v115) = v34 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v114, 0);
        v32 = v109[0];
      }
      if ( !v32 )
        goto LABEL_40;
      goto LABEL_39;
    }
  }
LABEL_27:
  nullsub_61();
  v129 = &unk_49DA100;
  nullsub_63();
  if ( v114 != (unsigned int *)v116 )
    _libc_free((unsigned __int64)v114);
}
