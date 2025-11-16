// Function: sub_1F2D8B0
// Address: 0x1f2d8b0
//
__int64 __fastcall sub_1F2D8B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 (*v3)(); // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rax
  _QWORD *v8; // rax
  unsigned __int8 *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  unsigned __int64 *v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 *v15; // r8
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int16 v19; // cx
  _QWORD *v20; // r15
  __int64 v21; // rdi
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // r12
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned __int8 *v29; // rsi
  _QWORD *v30; // rax
  unsigned int v31; // edx
  _QWORD *v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rax
  __int64 *v35; // r8
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // rax
  char v40; // r12
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rsi
  unsigned int v46; // ecx
  _QWORD *v47; // rdx
  _QWORD *v48; // r9
  _QWORD *v49; // rax
  __int64 v50; // rax
  __int64 *v51; // rsi
  __int64 v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // r15
  __int64 *v55; // rbx
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rsi
  unsigned __int8 *v59; // rsi
  _QWORD *v60; // r12
  unsigned int v61; // r15d
  unsigned int v62; // ebx
  __int64 v63; // rax
  __int64 v64; // r15
  _QWORD *v65; // rax
  _QWORD *v66; // rbx
  unsigned __int64 *v67; // r12
  __int64 v68; // rax
  unsigned __int64 v69; // rcx
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  _QWORD *v72; // rax
  __int64 v73; // rbx
  _QWORD **v74; // rax
  __int64 *v75; // rax
  __int64 v76; // rsi
  unsigned __int64 *v77; // r15
  __int64 v78; // rax
  unsigned __int64 v79; // rcx
  __int64 v80; // rsi
  unsigned __int8 *v81; // rsi
  int v82; // edx
  int v83; // r10d
  _QWORD *v84; // [rsp+8h] [rbp-118h]
  _QWORD *v85; // [rsp+8h] [rbp-118h]
  unsigned int v86; // [rsp+10h] [rbp-110h]
  unsigned __int64 *v87; // [rsp+10h] [rbp-110h]
  int v88; // [rsp+10h] [rbp-110h]
  __int64 v89; // [rsp+10h] [rbp-110h]
  __int64 v90; // [rsp+18h] [rbp-108h]
  _QWORD *v91; // [rsp+18h] [rbp-108h]
  char v92; // [rsp+27h] [rbp-F9h]
  __int64 v93; // [rsp+28h] [rbp-F8h]
  __int64 v94; // [rsp+30h] [rbp-F0h]
  __int64 v95; // [rsp+30h] [rbp-F0h]
  _QWORD *v96; // [rsp+38h] [rbp-E8h]
  __int64 v97; // [rsp+48h] [rbp-D8h]
  char v98; // [rsp+57h] [rbp-C9h] BYREF
  const char *v99; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v100[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v101; // [rsp+70h] [rbp-B0h]
  unsigned __int8 *v102[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int16 v103; // [rsp+90h] [rbp-90h]
  const char *v104; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v105; // [rsp+A8h] [rbp-78h]
  unsigned __int64 *v106; // [rsp+B0h] [rbp-70h]
  __int64 v107; // [rsp+B8h] [rbp-68h]
  __int64 v108; // [rsp+C0h] [rbp-60h]
  int v109; // [rsp+C8h] [rbp-58h]
  __int64 v110; // [rsp+D0h] [rbp-50h]
  __int64 v111; // [rsp+D8h] [rbp-48h]

  v3 = *(__int64 (**)())(**(_QWORD **)(a1 + 168) + 544LL);
  if ( v3 == sub_1F2AB40 || (v92 = v3()) == 0 )
  {
    v92 = byte_4FCAC60;
    if ( byte_4FCAC60 )
      v92 = ((*(_BYTE *)(*(_QWORD *)(a1 + 160) + 800LL) >> 1) ^ 1) & 1;
  }
  v4 = *(_QWORD *)(a1 + 232);
  v96 = 0;
  v93 = v4 + 72;
  v97 = *(_QWORD *)(v4 + 80);
  if ( v97 != v4 + 72 )
  {
    do
    {
      v20 = (_QWORD *)(v97 - 24);
      v94 = v97;
      v21 = v97 - 24;
      v97 = *(_QWORD *)(v97 + 8);
      v22 = sub_157EBA0(v21);
      if ( *(_BYTE *)(v22 + 16) != 25 )
        continue;
      if ( *(_BYTE *)(a1 + 464) )
      {
        if ( v92 )
          return *(unsigned __int8 *)(a1 + 464);
      }
      else
      {
        v23 = *(_QWORD *)(a1 + 168);
        *(_BYTE *)(a1 + 464) = 1;
        v98 = 0;
        v24 = *(__int64 **)(a1 + 240);
        v90 = v23;
        v25 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 80LL);
        if ( !v25 )
          BUG();
        v26 = *(_QWORD *)(v25 + 24);
        if ( !v26 )
        {
          v1 = sub_16498A0(0);
          v104 = 0;
          v106 = 0;
          v107 = v1;
          v108 = 0;
          v109 = 0;
          v110 = 0;
          v111 = 0;
          v105 = 0;
          BUG();
        }
        v27 = sub_16498A0(v26 - 24);
        v104 = 0;
        v107 = v27;
        v108 = 0;
        v109 = 0;
        v110 = 0;
        v111 = 0;
        v28 = *(_QWORD **)(v26 + 16);
        v106 = (unsigned __int64 *)v26;
        v105 = v28;
        v29 = *(unsigned __int8 **)(v26 + 24);
        v102[0] = v29;
        if ( v29 )
        {
          sub_1623A60((__int64)v102, (__int64)v29, 2);
          if ( v104 )
            sub_161E7C0((__int64)&v104, (__int64)v104);
          v104 = (const char *)v102[0];
          if ( v102[0] )
            sub_1623210((__int64)v102, v102[0], (__int64)&v104);
        }
        v30 = (_QWORD *)sub_16498A0(v22);
        v84 = (_QWORD *)sub_16471D0(v30, 0);
        v100[0] = (__int64)"StackGuardSlot";
        v101 = 259;
        v31 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(v105[7] + 40LL)) + 4);
        v103 = 257;
        v86 = v31;
        v32 = sub_1648A60(64, 1u);
        v96 = v32;
        if ( v32 )
          sub_15F8BC0((__int64)v32, v84, v86, 0, (__int64)v102, 0);
        if ( v105 )
        {
          v87 = v106;
          sub_157E9D0((__int64)(v105 + 5), (__int64)v96);
          v33 = *v87;
          v34 = v96[3];
          v96[4] = v87;
          v33 &= 0xFFFFFFFFFFFFFFF8LL;
          v96[3] = v33 | v34 & 7;
          *(_QWORD *)(v33 + 8) = v96 + 3;
          *v87 = *v87 & 7 | (unsigned __int64)(v96 + 3);
        }
        sub_164B780((__int64)v96, v100);
        v35 = v100;
        if ( v104 )
        {
          v99 = v104;
          sub_1623A60((__int64)&v99, (__int64)v104, 2);
          v35 = v100;
          v36 = v96[6];
          v37 = (__int64)(v96 + 6);
          if ( v36 )
          {
            sub_161E7C0((__int64)(v96 + 6), v36);
            v35 = v100;
            v37 = (__int64)(v96 + 6);
          }
          v38 = (unsigned __int8 *)v99;
          v96[6] = v99;
          if ( v38 )
          {
            sub_1623210((__int64)&v99, v38, v37);
            LODWORD(v35) = (unsigned int)v100;
          }
        }
        v88 = (int)v35;
        v100[0] = sub_1F2B5C0(v90, v24, (__int64 *)&v104, &v98);
        v103 = 257;
        v100[1] = (__int64)v96;
        v39 = sub_15E26F0(v24, 200, 0, 0);
        sub_1285290((__int64 *)&v104, *(_QWORD *)(v39 + 24), v39, v88, 2, (__int64)v102, 0);
        v40 = v98;
        if ( v104 )
          sub_161E7C0((__int64)&v104, (__int64)v104);
        v92 &= v40;
        if ( v92 )
          return *(unsigned __int8 *)(a1 + 464);
      }
      v5 = *(_QWORD *)(a1 + 168);
      *(_BYTE *)(a1 + 465) = 1;
      v6 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 552LL))(v5, *(_QWORD *)(a1 + 240));
      if ( v6 )
      {
        v7 = sub_16498A0(v22);
        v104 = 0;
        v107 = v7;
        v108 = 0;
        v109 = 0;
        v110 = 0;
        v111 = 0;
        v8 = *(_QWORD **)(v22 + 40);
        v106 = (unsigned __int64 *)(v22 + 24);
        v105 = v8;
        v9 = *(unsigned __int8 **)(v22 + 48);
        v102[0] = v9;
        if ( v9 )
        {
          sub_1623A60((__int64)v102, (__int64)v9, 2);
          if ( v104 )
            sub_161E7C0((__int64)&v104, (__int64)v104);
          v104 = (const char *)v102[0];
          if ( v102[0] )
            sub_1623210((__int64)v102, v102[0], (__int64)&v104);
        }
        v102[0] = (unsigned __int8 *)"Guard";
        v103 = 259;
        v10 = sub_1648A60(64, 1u);
        v11 = v10;
        if ( v10 )
          sub_15F9210((__int64)v10, *(_QWORD *)(*v96 + 24LL), (__int64)v96, 0, 1u, 0);
        if ( v105 )
        {
          v12 = v106;
          sub_157E9D0((__int64)(v105 + 5), (__int64)v11);
          v13 = v11[3];
          v14 = *v12;
          v11[4] = v12;
          v14 &= 0xFFFFFFFFFFFFFFF8LL;
          v11[3] = v14 | v13 & 7;
          *(_QWORD *)(v14 + 8) = v11 + 3;
          *v12 = *v12 & 7 | (unsigned __int64)(v11 + 3);
        }
        sub_164B780((__int64)v11, (__int64 *)v102);
        v15 = v100;
        if ( v104 )
        {
          v100[0] = (__int64)v104;
          sub_1623A60((__int64)v100, (__int64)v104, 2);
          v16 = v11[6];
          v15 = v100;
          if ( v16 )
          {
            sub_161E7C0((__int64)(v11 + 6), v16);
            v15 = v100;
          }
          v17 = (unsigned __int8 *)v100[0];
          v11[6] = v100[0];
          if ( v17 )
          {
            sub_1623210((__int64)v100, v17, (__int64)(v11 + 6));
            LODWORD(v15) = (unsigned int)v100;
          }
        }
        v100[0] = (__int64)v11;
        v103 = 257;
        v18 = sub_1285290((__int64 *)&v104, *(_QWORD *)(*(_QWORD *)v6 + 24LL), v6, (int)v15, 1, (__int64)v102, 0);
        v19 = *(_WORD *)(v18 + 18);
        *(_QWORD *)(v18 + 56) = *(_QWORD *)(v6 + 112);
        *(_WORD *)(v18 + 18) = v19 & 0x8000 | v19 & 3 | (*(_WORD *)(v6 + 18) >> 2) & 0xFFC;
        if ( !v104 )
          continue;
        goto LABEL_23;
      }
      v89 = sub_1F2B760(a1);
      v104 = "SP_return";
      LOWORD(v106) = 259;
      v42 = sub_157FBF0(v20, (__int64 *)(v22 + 24), (__int64)&v104);
      v43 = *(_QWORD *)(a1 + 248);
      v91 = (_QWORD *)v42;
      if ( v43 )
      {
        v44 = *(unsigned int *)(v43 + 48);
        if ( (_DWORD)v44 )
        {
          v45 = *(_QWORD *)(v43 + 32);
          v46 = (v44 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v47 = (_QWORD *)(v45 + 16LL * v46);
          v48 = (_QWORD *)*v47;
          if ( v20 == (_QWORD *)*v47 )
          {
LABEL_50:
            if ( v47 != (_QWORD *)(v45 + 16 * v44) && v47[1] )
            {
              sub_1F2D4D0(v43, (__int64)v91, (__int64)v20);
              sub_1F2D4D0(*(_QWORD *)(a1 + 248), v89, (__int64)v20);
            }
          }
          else
          {
            v82 = 1;
            while ( v48 != (_QWORD *)-8LL )
            {
              v83 = v82 + 1;
              v46 = (v44 - 1) & (v82 + v46);
              v47 = (_QWORD *)(v45 + 16LL * v46);
              v48 = (_QWORD *)*v47;
              if ( v20 == (_QWORD *)*v47 )
                goto LABEL_50;
              v82 = v83;
            }
          }
        }
      }
      v49 = (_QWORD *)sub_157EBA0((__int64)v20);
      sub_15F20C0(v49);
      sub_1580AC0(v91, (__int64)v20);
      v50 = sub_157E9C0((__int64)v20);
      v51 = *(__int64 **)(a1 + 240);
      v52 = *(_QWORD *)(a1 + 168);
      v107 = v50;
      v105 = v20;
      v104 = 0;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v111 = 0;
      v106 = (unsigned __int64 *)(v94 + 16);
      v95 = sub_1F2B5C0(v52, v51, (__int64 *)&v104, 0);
      v103 = 257;
      v53 = sub_1648A60(64, 1u);
      v54 = (__int64)v53;
      if ( v53 )
        sub_15F9210((__int64)v53, *(_QWORD *)(*v96 + 24LL), (__int64)v96, 0, 1u, 0);
      if ( v105 )
      {
        v55 = (__int64 *)v106;
        sub_157E9D0((__int64)(v105 + 5), v54);
        v56 = *(_QWORD *)(v54 + 24);
        v57 = *v55;
        *(_QWORD *)(v54 + 32) = v55;
        v57 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v54 + 24) = v57 | v56 & 7;
        *(_QWORD *)(v57 + 8) = v54 + 24;
        *v55 = *v55 & 7 | (v54 + 24);
      }
      sub_164B780(v54, (__int64 *)v102);
      if ( v104 )
      {
        v100[0] = (__int64)v104;
        sub_1623A60((__int64)v100, (__int64)v104, 2);
        v58 = *(_QWORD *)(v54 + 48);
        if ( v58 )
          sub_161E7C0(v54 + 48, v58);
        v59 = (unsigned __int8 *)v100[0];
        *(_QWORD *)(v54 + 48) = v100[0];
        if ( v59 )
          sub_1623210((__int64)v100, v59, v54 + 48);
      }
      v101 = 257;
      if ( *(_BYTE *)(v95 + 16) > 0x10u || *(_BYTE *)(v54 + 16) > 0x10u )
      {
        v103 = 257;
        v72 = sub_1648A60(56, 2u);
        v60 = v72;
        if ( v72 )
        {
          v73 = (__int64)v72;
          v74 = *(_QWORD ***)v95;
          if ( *(_BYTE *)(*(_QWORD *)v95 + 8LL) == 16 )
          {
            v85 = v74[4];
            v75 = (__int64 *)sub_1643320(*v74);
            v76 = (__int64)sub_16463B0(v75, (unsigned int)v85);
          }
          else
          {
            v76 = sub_1643320(*v74);
          }
          sub_15FEC10((__int64)v60, v76, 51, 32, v95, v54, (__int64)v102, 0);
        }
        else
        {
          v73 = 0;
        }
        if ( v105 )
        {
          v77 = v106;
          sub_157E9D0((__int64)(v105 + 5), (__int64)v60);
          v78 = v60[3];
          v79 = *v77;
          v60[4] = v77;
          v79 &= 0xFFFFFFFFFFFFFFF8LL;
          v60[3] = v79 | v78 & 7;
          *(_QWORD *)(v79 + 8) = v60 + 3;
          *v77 = *v77 & 7 | (unsigned __int64)(v60 + 3);
        }
        sub_164B780(v73, v100);
        if ( v104 )
        {
          v99 = v104;
          sub_1623A60((__int64)&v99, (__int64)v104, 2);
          v80 = v60[6];
          if ( v80 )
            sub_161E7C0((__int64)(v60 + 6), v80);
          v81 = (unsigned __int8 *)v99;
          v60[6] = v99;
          if ( v81 )
          {
            sub_1623210((__int64)&v99, v81, (__int64)(v60 + 6));
            if ( byte_4FCA2E8[0] )
              goto LABEL_66;
            goto LABEL_91;
          }
        }
      }
      else
      {
        v60 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v95, (_QWORD *)v54, 0);
      }
      if ( byte_4FCA2E8[0] )
        goto LABEL_66;
LABEL_91:
      if ( (unsigned int)sub_2207590(byte_4FCA2E8) )
      {
        sub_16AF710(dword_4FCA2F0, 0xFFFFFu, 0x100000u);
        sub_2207640(byte_4FCA2E8);
        v61 = dword_4FCA2F0[0];
        if ( byte_4FCA2E8[0] )
          goto LABEL_67;
        goto LABEL_93;
      }
LABEL_66:
      v61 = dword_4FCA2F0[0];
      if ( byte_4FCA2E8[0] )
        goto LABEL_67;
LABEL_93:
      if ( (unsigned int)sub_2207590(byte_4FCA2E8) )
      {
        sub_16AF710(dword_4FCA2F0, 0xFFFFFu, 0x100000u);
        sub_2207640(byte_4FCA2E8);
      }
LABEL_67:
      v62 = 0x80000000 - dword_4FCA2F0[0];
      v102[0] = (unsigned __int8 *)sub_15E0530(*(_QWORD *)(a1 + 232));
      v63 = sub_161BE60(v102, v61, v62);
      v103 = 257;
      v64 = v63;
      v65 = sub_1648A60(56, 3u);
      v66 = v65;
      if ( v65 )
        sub_15F83E0((__int64)v65, (__int64)v91, v89, (__int64)v60, 0);
      if ( v64 )
        sub_1625C10((__int64)v66, 2, v64);
      if ( v105 )
      {
        v67 = v106;
        sub_157E9D0((__int64)(v105 + 5), (__int64)v66);
        v68 = v66[3];
        v69 = *v67;
        v66[4] = v67;
        v69 &= 0xFFFFFFFFFFFFFFF8LL;
        v66[3] = v69 | v68 & 7;
        *(_QWORD *)(v69 + 8) = v66 + 3;
        *v67 = *v67 & 7 | (unsigned __int64)(v66 + 3);
      }
      sub_164B780((__int64)v66, (__int64 *)v102);
      if ( !v104 )
        continue;
      v100[0] = (__int64)v104;
      sub_1623A60((__int64)v100, (__int64)v104, 2);
      v70 = v66[6];
      if ( v70 )
        sub_161E7C0((__int64)(v66 + 6), v70);
      v71 = (unsigned __int8 *)v100[0];
      v66[6] = v100[0];
      if ( v71 )
        sub_1623210((__int64)v100, v71, (__int64)(v66 + 6));
      if ( !v104 )
        continue;
LABEL_23:
      sub_161E7C0((__int64)&v104, (__int64)v104);
    }
    while ( v93 != v97 );
  }
  return *(unsigned __int8 *)(a1 + 464);
}
