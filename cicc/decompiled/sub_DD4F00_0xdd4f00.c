// Function: sub_DD4F00
// Address: 0xdd4f00
//
__int64 __fastcall sub_DD4F00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  char v4; // di
  int v5; // edi
  _QWORD *v6; // rsi
  int v7; // r9d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // eax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // r12d
  bool v20; // al
  __int64 v21; // rsi
  __int64 v22; // r15
  _BYTE *v23; // r14
  _BYTE *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  _QWORD *v27; // r9
  _QWORD *v28; // r15
  _QWORD **v29; // r14
  _QWORD **v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rcx
  _QWORD *v36; // rdi
  unsigned __int64 v37; // rax
  _QWORD *v38; // r9
  _QWORD *v39; // r15
  _QWORD **v40; // r14
  __int64 v41; // rax
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  _QWORD *v45; // rax
  __int64 v46; // r14
  __int64 v47; // rax
  _QWORD *v48; // r9
  _QWORD *v49; // r15
  _QWORD **v50; // r14
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 *v55; // rax
  _QWORD *v56; // r9
  _QWORD *v57; // r15
  _QWORD **v58; // r14
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  _QWORD *v64; // r9
  _QWORD *v65; // r15
  _QWORD **v66; // r14
  __int64 v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // rcx
  unsigned __int64 v72; // rax
  _QWORD *v73; // r9
  _QWORD *v74; // r15
  _QWORD **v75; // r14
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // rdx
  _QWORD *v80; // rax
  __int64 v81; // rsi
  _QWORD *v82; // r9
  _QWORD *v83; // r15
  _QWORD **v84; // r14
  __int64 v85; // rax
  __int64 v86; // r9
  __int64 v87; // rdx
  char v88; // dl
  _QWORD *v89; // rax
  _QWORD *v90; // r9
  _QWORD *v91; // r15
  _QWORD **v92; // r14
  __int64 v93; // rax
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rdx
  __int64 v97; // rcx
  unsigned __int64 v98; // rax
  __int64 v99; // rsi
  __int64 v100; // rsi
  int v101; // r10d
  __int64 v102; // [rsp+8h] [rbp-B8h]
  __int64 v103; // [rsp+8h] [rbp-B8h]
  __int64 v104; // [rsp+8h] [rbp-B8h]
  __int64 v105; // [rsp+8h] [rbp-B8h]
  __int64 v106; // [rsp+8h] [rbp-B8h]
  __int64 v107; // [rsp+8h] [rbp-B8h]
  __int64 v108; // [rsp+8h] [rbp-B8h]
  __int64 v109; // [rsp+8h] [rbp-B8h]
  _QWORD *v110; // [rsp+10h] [rbp-B0h]
  _QWORD *v111; // [rsp+10h] [rbp-B0h]
  _QWORD *v112; // [rsp+10h] [rbp-B0h]
  _QWORD *v113; // [rsp+10h] [rbp-B0h]
  _QWORD *v114; // [rsp+10h] [rbp-B0h]
  _QWORD *v115; // [rsp+10h] [rbp-B0h]
  _QWORD *v116; // [rsp+10h] [rbp-B0h]
  _QWORD *v117; // [rsp+10h] [rbp-B0h]
  char v118; // [rsp+1Fh] [rbp-A1h]
  char v119; // [rsp+1Fh] [rbp-A1h]
  char v120; // [rsp+1Fh] [rbp-A1h]
  char v121; // [rsp+1Fh] [rbp-A1h]
  char v122; // [rsp+1Fh] [rbp-A1h]
  char v123; // [rsp+1Fh] [rbp-A1h]
  char v124; // [rsp+1Fh] [rbp-A1h]
  char v125; // [rsp+1Fh] [rbp-A1h]
  __int64 v126; // [rsp+28h] [rbp-98h] BYREF
  __int64 v127[5]; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v128; // [rsp+60h] [rbp-60h] BYREF
  __int64 v129; // [rsp+68h] [rbp-58h]
  _QWORD v130[10]; // [rsp+70h] [rbp-50h] BYREF

  v2 = a2;
  v4 = *((_BYTE *)a1 + 16);
  v126 = a2;
  v5 = v4 & 1;
  if ( v5 )
  {
    v6 = a1 + 3;
    v7 = 3;
  }
  else
  {
    v13 = *((unsigned int *)a1 + 8);
    v6 = (_QWORD *)a1[3];
    if ( !(_DWORD)v13 )
      goto LABEL_12;
    v7 = v13 - 1;
  }
  v8 = v7 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v9 = &v6[2 * v8];
  v10 = *v9;
  if ( v2 == *v9 )
    goto LABEL_4;
  v15 = 1;
  while ( v10 != -4096 )
  {
    v101 = v15 + 1;
    v8 = v7 & (v15 + v8);
    v9 = &v6[2 * v8];
    v10 = *v9;
    if ( v2 == *v9 )
      goto LABEL_4;
    v15 = v101;
  }
  if ( (_BYTE)v5 )
  {
    v14 = 8;
    goto LABEL_13;
  }
  v13 = *((unsigned int *)a1 + 8);
LABEL_12:
  v14 = 2 * v13;
LABEL_13:
  v9 = &v6[v14];
LABEL_4:
  v11 = 8;
  if ( !(_BYTE)v5 )
    v11 = 2LL * *((unsigned int *)a1 + 8);
  if ( v9 == &v6[v11] )
  {
    switch ( *(_WORD *)(v2 + 24) )
    {
      case 0:
      case 1:
      case 0x10:
        goto LABEL_25;
      case 2:
        v100 = sub_DD4F00(a1, *(_QWORD *)(v2 + 32));
        if ( v100 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5200(*a1, v100, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_25;
      case 3:
        v99 = sub_DD4F00(a1, *(_QWORD *)(v2 + 32));
        if ( v99 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC2B70(*a1, v99, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_25;
      case 4:
        v81 = sub_DD4F00(a1, *(_QWORD *)(v2 + 32));
        if ( v81 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DC5000(*a1, v81, *(_QWORD *)(v2 + 40), 0);
        goto LABEL_25;
      case 5:
        v128 = v130;
        v129 = 0x200000000LL;
        v73 = *(_QWORD **)(v2 + 32);
        v115 = &v73[*(_QWORD *)(v2 + 40)];
        if ( v73 == v115 )
          goto LABEL_25;
        v123 = 0;
        v74 = *(_QWORD **)(v2 + 32);
        do
        {
          v75 = (_QWORD **)*v74;
          v30 = (_QWORD **)*v74;
          v76 = sub_DD4F00(a1, *v74);
          v79 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v109 = v76;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v77, v78);
            v79 = (unsigned int)v129;
            v76 = v109;
          }
          v128[v79] = v76;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v74;
          v123 |= v128[(unsigned int)v129 - 1] != (_QWORD)v75;
        }
        while ( v115 != v74 );
        if ( v123 )
        {
          v30 = &v128;
          v80 = sub_DC7EB0((__int64 *)*a1, (__int64)&v128, 0, 0);
          v36 = v128;
          v2 = (__int64)v80;
        }
        goto LABEL_63;
      case 6:
        v128 = v130;
        v129 = 0x200000000LL;
        v48 = *(_QWORD **)(v2 + 32);
        v112 = &v48[*(_QWORD *)(v2 + 40)];
        if ( v48 == v112 )
          goto LABEL_25;
        v120 = 0;
        v49 = *(_QWORD **)(v2 + 32);
        do
        {
          v50 = (_QWORD **)*v49;
          v30 = (_QWORD **)*v49;
          v51 = sub_DD4F00(a1, *v49);
          v54 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v106 = v51;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v52, v53);
            v54 = (unsigned int)v129;
            v51 = v106;
          }
          v128[v54] = v51;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v49;
          v120 |= v128[(unsigned int)v129 - 1] != (_QWORD)v50;
        }
        while ( v112 != v49 );
        if ( v120 )
        {
          v30 = &v128;
          v55 = sub_DC8BD0((__int64 *)*a1, (__int64)&v128, 0, 0);
          v36 = v128;
          v2 = (__int64)v55;
        }
        goto LABEL_63;
      case 7:
        v46 = sub_DD4F00(a1, *(_QWORD *)(v2 + 32));
        v47 = sub_DD4F00(a1, *(_QWORD *)(v2 + 40));
        if ( v46 != *(_QWORD *)(v2 + 32) || v47 != *(_QWORD *)(v2 + 40) )
          v2 = sub_DCB270(*a1, v46, v47);
        goto LABEL_25;
      case 8:
        v128 = v130;
        v129 = 0x200000000LL;
        v38 = *(_QWORD **)(v2 + 32);
        v111 = &v38[*(_QWORD *)(v2 + 40)];
        if ( v38 == v111 )
          goto LABEL_25;
        v119 = 0;
        v39 = *(_QWORD **)(v2 + 32);
        do
        {
          v40 = (_QWORD **)*v39;
          v30 = (_QWORD **)*v39;
          v41 = sub_DD4F00(a1, *v39);
          v44 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v105 = v41;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v42, v43);
            v44 = (unsigned int)v129;
            v41 = v105;
          }
          v128[v44] = v41;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v39;
          v119 |= v128[(unsigned int)v129 - 1] != (_QWORD)v40;
        }
        while ( v111 != v39 );
        if ( v119 )
        {
          v30 = &v128;
          v45 = sub_DBFF60(*a1, (unsigned int *)&v128, *(_QWORD *)(v2 + 48), *(_WORD *)(v2 + 28) & 7);
          v36 = v128;
          v2 = (__int64)v45;
        }
        goto LABEL_63;
      case 9:
        v128 = v130;
        v129 = 0x200000000LL;
        v27 = *(_QWORD **)(v2 + 32);
        v110 = &v27[*(_QWORD *)(v2 + 40)];
        if ( v27 == v110 )
          goto LABEL_25;
        v118 = 0;
        v28 = *(_QWORD **)(v2 + 32);
        do
        {
          v29 = (_QWORD **)*v28;
          v30 = (_QWORD **)*v28;
          v31 = sub_DD4F00(a1, *v28);
          v34 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v104 = v31;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v32, v33);
            v34 = (unsigned int)v129;
            v31 = v104;
          }
          v35 = (__int64)v128;
          v128[v34] = v31;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v28;
          v118 |= v128[(unsigned int)v129 - 1] != (_QWORD)v29;
        }
        while ( v110 != v28 );
        if ( v118 )
        {
          v30 = &v128;
          v37 = sub_DCE040((__int64 *)*a1, (__int64)&v128, v34, v35, v32);
          v36 = v128;
          v2 = v37;
        }
        goto LABEL_63;
      case 0xA:
        v128 = v130;
        v129 = 0x200000000LL;
        v90 = *(_QWORD **)(v2 + 32);
        v117 = &v90[*(_QWORD *)(v2 + 40)];
        if ( v90 == v117 )
          goto LABEL_25;
        v125 = 0;
        v91 = *(_QWORD **)(v2 + 32);
        do
        {
          v92 = (_QWORD **)*v91;
          v30 = (_QWORD **)*v91;
          v93 = sub_DD4F00(a1, *v91);
          v96 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v103 = v93;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v94, v95);
            v96 = (unsigned int)v129;
            v93 = v103;
          }
          v97 = (__int64)v128;
          v128[v96] = v93;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v91;
          v125 |= v128[(unsigned int)v129 - 1] != (_QWORD)v92;
        }
        while ( v117 != v91 );
        if ( v125 )
        {
          v30 = &v128;
          v98 = sub_DCDF90((__int64 *)*a1, (__int64)&v128, v96, v97, v94);
          v36 = v128;
          v2 = v98;
        }
        goto LABEL_63;
      case 0xB:
        v128 = v130;
        v129 = 0x200000000LL;
        v82 = *(_QWORD **)(v2 + 32);
        v116 = &v82[*(_QWORD *)(v2 + 40)];
        if ( v82 == v116 )
          goto LABEL_25;
        v124 = 0;
        v83 = *(_QWORD **)(v2 + 32);
        do
        {
          v84 = (_QWORD **)*v83;
          v30 = (_QWORD **)*v83;
          v85 = sub_DD4F00(a1, *v83);
          v87 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v107 = v85;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v60, v86);
            v87 = (unsigned int)v129;
            v85 = v107;
          }
          v63 = (__int64)v128;
          v128[v87] = v85;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v83;
          v124 |= v128[(unsigned int)v129 - 1] != (_QWORD)v84;
        }
        while ( v116 != v83 );
        v88 = 0;
        if ( v124 )
          goto LABEL_87;
        goto LABEL_63;
      case 0xC:
        v128 = v130;
        v129 = 0x200000000LL;
        v64 = *(_QWORD **)(v2 + 32);
        v114 = &v64[*(_QWORD *)(v2 + 40)];
        if ( v64 == v114 )
          goto LABEL_25;
        v122 = 0;
        v65 = *(_QWORD **)(v2 + 32);
        do
        {
          v66 = (_QWORD **)*v65;
          v30 = (_QWORD **)*v65;
          v67 = sub_DD4F00(a1, *v65);
          v70 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v102 = v67;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v68, v69);
            v70 = (unsigned int)v129;
            v67 = v102;
          }
          v71 = (__int64)v128;
          v128[v70] = v67;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v65;
          v122 |= v128[(unsigned int)v129 - 1] != (_QWORD)v66;
        }
        while ( v114 != v65 );
        if ( v122 )
        {
          v30 = &v128;
          v72 = sub_DCE150((__int64 *)*a1, (__int64)&v128, v70, v71, v68);
          v36 = v128;
          v2 = v72;
        }
        goto LABEL_63;
      case 0xD:
        v128 = v130;
        v129 = 0x200000000LL;
        v56 = *(_QWORD **)(v2 + 32);
        v113 = &v56[*(_QWORD *)(v2 + 40)];
        if ( v56 == v113 )
          goto LABEL_25;
        v121 = 0;
        v57 = *(_QWORD **)(v2 + 32);
        do
        {
          v58 = (_QWORD **)*v57;
          v30 = (_QWORD **)*v57;
          v59 = sub_DD4F00(a1, *v57);
          v62 = (unsigned int)v129;
          if ( (unsigned __int64)(unsigned int)v129 + 1 > HIDWORD(v129) )
          {
            v30 = (_QWORD **)v130;
            v108 = v59;
            sub_C8D5F0((__int64)&v128, v130, (unsigned int)v129 + 1LL, 8u, v60, v61);
            v62 = (unsigned int)v129;
            v59 = v108;
          }
          v63 = (__int64)v128;
          v128[v62] = v59;
          v36 = v128;
          LODWORD(v129) = v129 + 1;
          ++v57;
          v121 |= v128[(unsigned int)v129 - 1] != (_QWORD)v58;
        }
        while ( v113 != v57 );
        if ( !v121 )
          goto LABEL_63;
        v88 = 1;
LABEL_87:
        v30 = &v128;
        v89 = sub_DCEE50((__int64 *)*a1, (__int64)&v128, v88, v63, v60);
        v36 = v128;
        v2 = (__int64)v89;
LABEL_63:
        if ( v36 != v130 )
          _libc_free(v36, v30);
LABEL_25:
        v127[0] = v2;
        sub_DB11F0((__int64)&v128, (__int64)(a1 + 1), &v126, v127);
        v9 = (__int64 *)v130[0];
        break;
      case 0xE:
        v26 = sub_DD4F00(a1, *(_QWORD *)(v2 + 32));
        if ( v26 != *(_QWORD *)(v2 + 32) )
          v2 = (__int64)sub_DD3A70(*a1, v26, *(_QWORD *)(v2 + 40));
        goto LABEL_25;
      case 0xF:
        v22 = v2;
        if ( !sub_DADE90(*a1, v2, a1[11]) )
        {
          v23 = *(_BYTE **)(v2 - 8);
          v24 = (_BYTE *)a1[12];
          if ( *v23 == 86 )
          {
            if ( v24 == *((_BYTE **)v23 - 12) )
            {
              v16 = sub_DA36A0((__int64 *)*a1, *((_BYTE *)a1 + 104));
              v127[4] = v17;
              v127[3] = (__int64)v16;
              if ( (_BYTE)v17 )
              {
                v18 = v16[4];
                v19 = *(_DWORD *)(v18 + 32);
                if ( v19 <= 0x40 )
                  v20 = *(_QWORD *)(v18 + 24) == 1;
                else
                  v20 = v19 - 1 == (unsigned int)sub_C444A0(v18 + 24);
                if ( v20 )
                  v21 = *((_QWORD *)v23 - 8);
                else
                  v21 = *((_QWORD *)v23 - 4);
                v2 = sub_DD8400(*a1, v21);
              }
            }
          }
          else if ( v24 == v23 )
          {
            v2 = (__int64)sub_DA36A0((__int64 *)*a1, *((_BYTE *)a1 + 104));
            v127[1] = v2;
            v127[2] = v25;
            if ( !(_BYTE)v25 )
              v2 = v22;
          }
        }
        goto LABEL_25;
      default:
        BUG();
    }
  }
  return v9[1];
}
