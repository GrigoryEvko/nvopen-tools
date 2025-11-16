// Function: sub_24E6B60
// Address: 0x24e6b60
//
void __fastcall sub_24E6B60(unsigned __int8 *a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int8 *v4; // r15
  __int64 v6; // rax
  _QWORD *v7; // rsi
  int v8; // eax
  unsigned __int64 v9; // rax
  __int64 ***v10; // rax
  __int64 **v11; // r12
  __int64 v12; // rax
  _BYTE *v13; // r14
  __int64 v14; // rax
  __int64 v15; // r14
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  _BYTE *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rdi
  int v26; // eax
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // r12
  unsigned __int8 *v32; // rax
  __int64 v33; // rsi
  unsigned __int8 *v34; // r15
  int v35; // r13d
  _BYTE *v36; // rdx
  int v37; // eax
  __int64 *v38; // rax
  __int64 v39; // rax
  int v40; // eax
  __int64 *v41; // rax
  int v42; // eax
  __int64 *v43; // rax
  int v44; // edx
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r12
  int v49; // r12d
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r12
  __int64 v54; // rcx
  unsigned __int64 v55; // rax
  int v56; // edx
  unsigned __int64 v57; // rax
  __int64 *v58; // r13
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // r14
  __int64 *v61; // rsi
  __int64 *v62; // r8
  _QWORD *v63; // r15
  unsigned __int64 v64; // rax
  int v65; // edx
  _QWORD *v66; // rdi
  _QWORD *v67; // rax
  _QWORD *v68; // r14
  _QWORD *v69; // r13
  __int64 v70; // rax
  __int64 *v71; // rcx
  __int64 v74; // [rsp+20h] [rbp-320h]
  unsigned __int8 *v75; // [rsp+20h] [rbp-320h]
  _BYTE v76[32]; // [rsp+40h] [rbp-300h] BYREF
  __int16 v77; // [rsp+60h] [rbp-2E0h]
  _BYTE *v78; // [rsp+70h] [rbp-2D0h] BYREF
  __int64 v79; // [rsp+78h] [rbp-2C8h]
  _BYTE v80[32]; // [rsp+80h] [rbp-2C0h] BYREF
  __int64 v81; // [rsp+A0h] [rbp-2A0h]
  __int64 v82; // [rsp+A8h] [rbp-298h]
  __int64 v83; // [rsp+B0h] [rbp-290h]
  __int64 v84; // [rsp+B8h] [rbp-288h]
  void **v85; // [rsp+C0h] [rbp-280h]
  void **v86; // [rsp+C8h] [rbp-278h]
  __int64 v87; // [rsp+D0h] [rbp-270h]
  int v88; // [rsp+D8h] [rbp-268h]
  __int16 v89; // [rsp+DCh] [rbp-264h]
  char v90; // [rsp+DEh] [rbp-262h]
  __int64 v91; // [rsp+E0h] [rbp-260h]
  __int64 v92; // [rsp+E8h] [rbp-258h]
  void *v93; // [rsp+F0h] [rbp-250h] BYREF
  void *v94; // [rsp+F8h] [rbp-248h] BYREF
  __int64 v95[2]; // [rsp+100h] [rbp-240h] BYREF
  _BYTE v96[32]; // [rsp+110h] [rbp-230h] BYREF
  __int64 v97; // [rsp+130h] [rbp-210h]
  __int64 v98; // [rsp+138h] [rbp-208h]
  __int16 v99; // [rsp+140h] [rbp-200h]
  __int64 v100; // [rsp+148h] [rbp-1F8h]
  void **v101; // [rsp+150h] [rbp-1F0h]
  void **v102; // [rsp+158h] [rbp-1E8h]
  __int64 v103; // [rsp+160h] [rbp-1E0h]
  int v104; // [rsp+168h] [rbp-1D8h]
  __int16 v105; // [rsp+16Ch] [rbp-1D4h]
  char v106; // [rsp+16Eh] [rbp-1D2h]
  __int64 v107; // [rsp+170h] [rbp-1D0h]
  __int64 v108; // [rsp+178h] [rbp-1C8h]
  void *v109; // [rsp+180h] [rbp-1C0h] BYREF
  void *v110; // [rsp+188h] [rbp-1B8h] BYREF
  _QWORD v111[4]; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v112; // [rsp+1B0h] [rbp-190h]
  _BYTE *v113; // [rsp+1B8h] [rbp-188h]
  __int64 v114; // [rsp+1C0h] [rbp-180h]
  _BYTE v115[32]; // [rsp+1C8h] [rbp-178h] BYREF
  _BYTE *v116; // [rsp+1E8h] [rbp-158h]
  __int64 v117; // [rsp+1F0h] [rbp-150h]
  _BYTE v118[192]; // [rsp+1F8h] [rbp-148h] BYREF
  _BYTE *v119; // [rsp+2B8h] [rbp-88h]
  __int64 v120; // [rsp+2C0h] [rbp-80h]
  _BYTE v121[120]; // [rsp+2C8h] [rbp-78h] BYREF

  v4 = a1;
  v6 = sub_BD5C60((__int64)a1);
  LOWORD(v83) = 0;
  v84 = v6;
  v85 = &v93;
  v86 = &v94;
  v79 = 0x200000000LL;
  v7 = a1;
  v89 = 512;
  v78 = v80;
  v87 = 0;
  v88 = 0;
  v90 = 7;
  v91 = 0;
  v92 = 0;
  v81 = 0;
  v82 = 0;
  v93 = &unk_49DA100;
  v94 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v78, (__int64)a1);
  v8 = *(_DWORD *)(a2 + 280);
  if ( v8 == 2 )
  {
    if ( !*(_BYTE *)(a2 + 360) )
    {
      sub_24F4CC0(a2, &v78, a3, 0);
      v40 = *(_DWORD *)(a2 + 280);
      if ( !v40 )
      {
        v111[0] = sub_BCE3C0(**(__int64 ***)(a2 + 288), 0);
        v41 = (__int64 *)sub_BCB120(**(_QWORD ***)(a2 + 288));
        v28 = sub_BCF480(v41, v111, 1, 0);
LABEL_32:
        v29 = **(_QWORD **)(v28 + 16);
        v30 = 32 * (2LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF));
        v31 = *(_QWORD *)&a1[v30];
        if ( *(_BYTE *)v31 == 21 )
          goto LABEL_44;
        v32 = sub_24E54B0(*(unsigned __int8 **)&a1[v30]);
        if ( *(_BYTE *)(v29 + 8) == 15 )
        {
          v33 = sub_ACADE0((__int64 **)v29);
          v75 = sub_24E54B0((unsigned __int8 *)v31);
          if ( v75 != (unsigned __int8 *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF)) )
          {
            v34 = (unsigned __int8 *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
            v35 = 0;
            do
            {
              v36 = *(_BYTE **)v34;
              v37 = v35;
              LOWORD(v112) = 257;
              ++v35;
              LODWORD(v95[0]) = v37;
              v34 += 32;
              v33 = sub_2466140((__int64 *)&v78, v33, v36, v95, 1, (__int64)v111);
            }
            while ( v75 != v34 );
            v4 = a1;
          }
        }
        else
        {
          v71 = (__int64 *)(v31 - 32LL * (*(_DWORD *)(v31 + 4) & 0x7FFFFFF));
          if ( !(unsigned int)((v32 - (unsigned __int8 *)v71) >> 5) )
          {
            sub_24E5740((__int64 *)&v78);
            goto LABEL_39;
          }
          v33 = *v71;
        }
        sub_24E57E0((__int64 *)&v78, v33);
LABEL_39:
        v38 = (__int64 *)sub_BD5C60(v31);
        v39 = sub_AC3540(v38);
        sub_BD84D0(v31, v39);
        sub_B43D60((_QWORD *)v31);
        goto LABEL_21;
      }
      if ( v40 < 0 )
        goto LABEL_105;
      if ( v40 > 2 )
      {
        if ( v40 == 3 )
          BUG();
        goto LABEL_105;
      }
    }
    v28 = *(_QWORD *)(*(_QWORD *)(a2 + 328) + 24LL);
    goto LABEL_32;
  }
  if ( v8 > 2 )
  {
    if ( v8 != 3 )
      goto LABEL_21;
    v100 = sub_BD5C60((__int64)a1);
    v101 = &v109;
    v102 = &v110;
    v95[0] = (__int64)v96;
    v95[1] = 0x200000000LL;
    v103 = 0;
    v104 = 0;
    v105 = 512;
    v106 = 7;
    v107 = 0;
    v108 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v109 = &unk_49DA100;
    v110 = &unk_49DA0B0;
    sub_D5F1F0((__int64)v95, (__int64)a1);
    v22 = *((_QWORD *)a1 - 4);
    if ( !v22 || *(_BYTE *)v22 || *(_QWORD *)(v22 + 24) != *((_QWORD *)a1 + 10) )
      BUG();
    if ( *(_DWORD *)(v22 + 36) != 44 )
      goto LABEL_19;
    v44 = *a1;
    if ( v44 == 40 )
    {
      v45 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
    }
    else
    {
      v45 = 0;
      if ( v44 != 85 )
      {
        v45 = 64;
        if ( v44 != 34 )
          goto LABEL_105;
      }
    }
    if ( (a1[7] & 0x80u) != 0 )
    {
      v46 = sub_BD2BC0((__int64)a1);
      v48 = v46 + v47;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v48 >> 4) )
          goto LABEL_111;
      }
      else if ( (unsigned int)((v48 - sub_BD2BC0((__int64)a1)) >> 4) )
      {
        if ( (a1[7] & 0x80u) != 0 )
        {
          v49 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
          if ( (a1[7] & 0x80u) == 0 )
            BUG();
          v50 = sub_BD2BC0((__int64)a1);
          v52 = 32LL * (unsigned int)(*(_DWORD *)(v50 + v51 - 4) - v49);
LABEL_57:
          if ( (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v45 - v52) >> 5) > 2
            && sub_BD3990(*(unsigned __int8 **)&a1[32 * (2LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))], (__int64)a1) )
          {
            v53 = *((_QWORD *)a1 + 5);
            v54 = sub_AA54C0(v53);
            v55 = *(_QWORD *)(v54 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v55 == v54 + 48 )
            {
              v57 = 0;
            }
            else
            {
              if ( !v55 )
                BUG();
              v56 = *(unsigned __int8 *)(v55 - 24);
              v57 = v55 - 24;
              if ( (unsigned int)(v56 - 30) >= 0xB )
                v57 = 0;
            }
            v58 = (__int64 *)(a1 + 24);
            v59 = *(_QWORD *)(v57 + 24) & 0xFFFFFFFFFFFFFFF8LL;
            v60 = v59 - 24;
            if ( !v59 )
              v60 = 0;
            v61 = *(__int64 **)(v60 + 32);
            v62 = (__int64 *)(v60 + 24);
            if ( v58 != v61 && v62 != v58 )
              sub_AA80F0(v53, (unsigned __int64 *)a1 + 3, 0, v54, v62, 0, v61, 0);
            sub_D5F1F0((__int64)v95, (__int64)a1);
            sub_24E5740(v95);
            v63 = (_QWORD *)*((_QWORD *)a1 + 5);
            v113 = v115;
            v114 = 0x400000000LL;
            v119 = v121;
            v77 = 257;
            memset(v111, 0, sizeof(v111));
            v112 = 0;
            v116 = v118;
            v117 = 0x800000000LL;
            v120 = 0x800000000LL;
            v121[64] = 1;
            sub_AA8550(v63, v58, 0, (__int64)v76, 0);
            v64 = v63[6] & 0xFFFFFFFFFFFFFFF8LL;
            if ( (_QWORD *)v64 == v63 + 6 )
            {
              v66 = 0;
            }
            else
            {
              if ( !v64 )
                BUG();
              v65 = *(unsigned __int8 *)(v64 - 24);
              v66 = 0;
              v67 = (_QWORD *)(v64 - 24);
              if ( (unsigned int)(v65 - 30) < 0xB )
                v66 = v67;
            }
            sub_B43D60(v66);
            v7 = v111;
            sub_29F2700(v60, v111, 0, 0, 1, 0);
            if ( v119 != v121 )
              _libc_free((unsigned __int64)v119);
            v68 = v116;
            v69 = &v116[24 * (unsigned int)v117];
            if ( v116 != (_BYTE *)v69 )
            {
              do
              {
                v70 = *(v69 - 1);
                v69 -= 3;
                if ( v70 != -4096 && v70 != 0 && v70 != -8192 )
                  sub_BD60C0(v69);
              }
              while ( v68 != v69 );
              v69 = v116;
            }
            if ( v69 != (_QWORD *)v118 )
              _libc_free((unsigned __int64)v69);
            if ( v113 != v115 )
              _libc_free((unsigned __int64)v113);
            nullsub_61();
            v109 = &unk_49DA100;
            nullsub_63();
            if ( (_BYTE *)v95[0] != v96 )
              _libc_free(v95[0]);
            goto LABEL_29;
          }
LABEL_19:
          sub_24E5740(v95);
          nullsub_61();
          v109 = &unk_49DA100;
          nullsub_63();
          if ( (_BYTE *)v95[0] != v96 )
            _libc_free(v95[0]);
          goto LABEL_21;
        }
LABEL_111:
        BUG();
      }
    }
    v52 = 0;
    goto LABEL_57;
  }
  if ( !v8 )
  {
    if ( !a4 )
    {
LABEL_29:
      sub_F94A20(&v78, (__int64)v7);
      return;
    }
LABEL_44:
    sub_24E5740((__int64 *)&v78);
    goto LABEL_21;
  }
  if ( v8 == 1 )
  {
    if ( *(_BYTE *)(a2 + 360) )
    {
LABEL_6:
      v9 = *(_QWORD *)(*(_QWORD *)(a2 + 328) + 24LL);
LABEL_7:
      v10 = *(__int64 ****)(v9 + 16);
      v11 = *v10;
      if ( *((_BYTE *)*v10 + 8) == 15 )
      {
        v12 = sub_AC9EC0((__int64 **)*v11[2]);
        LODWORD(v95[0]) = 0;
        v13 = (_BYTE *)v12;
        LOWORD(v112) = 257;
        v14 = sub_ACADE0(v11);
        v15 = sub_2466140((__int64 *)&v78, v14, v13, v95, 1, (__int64)v111);
      }
      else
      {
        v15 = sub_AC9EC0(*v10);
      }
      LOWORD(v112) = 257;
      v74 = v84;
      v16 = sub_BD2C40(72, v15 != 0);
      v17 = (__int64)v16;
      if ( v16 )
        sub_B4BB80((__int64)v16, v74, v15, v15 != 0, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v86 + 2))(v86, v17, v111, v82, v83);
      v18 = (__int64)v78;
      v19 = &v78[16 * (unsigned int)v79];
      if ( v78 != v19 )
      {
        do
        {
          v20 = *(_QWORD *)(v18 + 8);
          v21 = *(_DWORD *)v18;
          v18 += 16;
          sub_B99FD0(v17, v21, v20);
        }
        while ( v19 != (_BYTE *)v18 );
      }
      goto LABEL_21;
    }
    sub_24F4CC0(a2, &v78, a3, 0);
    v42 = *(_DWORD *)(a2 + 280);
    if ( !v42 )
    {
      v111[0] = sub_BCE3C0(**(__int64 ***)(a2 + 288), 0);
      v43 = (__int64 *)sub_BCB120(**(_QWORD ***)(a2 + 288));
      v9 = sub_BCF480(v43, v111, 1, 0);
      goto LABEL_7;
    }
    if ( v42 >= 0 )
    {
      if ( v42 <= 2 )
        goto LABEL_6;
      if ( v42 == 3 )
        BUG();
    }
LABEL_105:
    BUG();
  }
LABEL_21:
  v23 = (_QWORD *)*((_QWORD *)v4 + 5);
  LOWORD(v112) = 257;
  v24 = v23;
  v23 += 6;
  sub_AA8550(v24, (__int64 *)v4 + 3, 0, (__int64)v111, 0);
  v25 = *v23 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v25 == v23 )
  {
    v27 = 0;
  }
  else
  {
    if ( !v25 )
      BUG();
    v26 = *(unsigned __int8 *)(v25 - 24);
    v27 = (_QWORD *)(v25 - 24);
    if ( (unsigned int)(v26 - 30) >= 0xB )
      v27 = 0;
  }
  sub_B43D60(v27);
  nullsub_61();
  v93 = &unk_49DA100;
  nullsub_63();
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
}
