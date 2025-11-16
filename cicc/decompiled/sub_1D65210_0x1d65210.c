// Function: sub_1D65210
// Address: 0x1d65210
//
__int64 __fastcall sub_1D65210(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r13
  __int64 i; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rcx
  _QWORD *v19; // rdx
  __int64 v20; // r14
  __int64 *v21; // r15
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // edi
  _QWORD *v26; // rdx
  __int64 v27; // r8
  _QWORD *v28; // rcx
  unsigned __int64 v29; // r9
  unsigned __int8 v30; // dl
  unsigned __int64 v31; // r8
  unsigned __int8 v32; // cl
  unsigned __int64 v33; // r11
  unsigned __int64 v34; // r10
  _QWORD *v35; // rdx
  __int64 *v36; // r10
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  __int64 v39; // rdx
  _QWORD *v40; // rax
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned __int64 v45; // rdx
  int v46; // r8d
  unsigned int v47; // esi
  __int64 v48; // r15
  __int64 v49; // r14
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 v52; // r13
  __int64 *v53; // rax
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // r9
  __int64 *v58; // r10
  __int64 v59; // rsi
  __int64 v60; // r9
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // r11
  __int64 v66; // r10
  __int64 v67; // rax
  __int64 *v68; // rax
  __int64 *v69; // rax
  __int64 v70; // r10
  __int64 *v71; // r11
  __int64 *v72; // rdi
  __int64 *v73; // rcx
  __int64 *v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // rax
  unsigned __int64 v77; // rsi
  __int64 v78; // rax
  double v79; // xmm4_8
  double v80; // xmm5_8
  __int64 v81; // rax
  __int64 v82; // rax
  unsigned __int64 v84; // r8
  unsigned __int64 v85; // r9
  __int64 v86; // rax
  unsigned __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 *v90; // rax
  int v91; // [rsp+Ch] [rbp-144h]
  __int64 *v92; // [rsp+10h] [rbp-140h]
  __int64 v93; // [rsp+10h] [rbp-140h]
  __int64 v94; // [rsp+10h] [rbp-140h]
  __int64 v95; // [rsp+18h] [rbp-138h]
  __int64 *v96; // [rsp+18h] [rbp-138h]
  __int64 *v97; // [rsp+18h] [rbp-138h]
  __int64 v98; // [rsp+20h] [rbp-130h]
  __int64 v99; // [rsp+20h] [rbp-130h]
  __int64 **v100; // [rsp+28h] [rbp-128h]
  __int64 v101; // [rsp+28h] [rbp-128h]
  unsigned int v102; // [rsp+30h] [rbp-120h]
  __int64 *v103; // [rsp+48h] [rbp-108h]
  __int64 v104; // [rsp+48h] [rbp-108h]
  __int64 *v105; // [rsp+48h] [rbp-108h]
  __int64 *v106; // [rsp+50h] [rbp-100h]
  _QWORD *v107; // [rsp+58h] [rbp-F8h]
  __int64 *v108; // [rsp+58h] [rbp-F8h]
  __int64 *v109; // [rsp+58h] [rbp-F8h]
  __int64 *v110; // [rsp+58h] [rbp-F8h]
  _BYTE *v111; // [rsp+58h] [rbp-F8h]
  unsigned __int64 *v112; // [rsp+58h] [rbp-F8h]
  __int64 *v113; // [rsp+58h] [rbp-F8h]
  unsigned __int8 v114; // [rsp+68h] [rbp-E8h]
  __int64 *v115; // [rsp+68h] [rbp-E8h]
  __int64 v116; // [rsp+68h] [rbp-E8h]
  __int64 v117; // [rsp+68h] [rbp-E8h]
  __int64 v118; // [rsp+68h] [rbp-E8h]
  __int64 v119; // [rsp+68h] [rbp-E8h]
  _QWORD *v120; // [rsp+68h] [rbp-E8h]
  __int64 v121; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v122; // [rsp+80h] [rbp-D0h]
  unsigned __int8 *v123[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v124; // [rsp+A0h] [rbp-B0h]
  __int64 **v125; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v126; // [rsp+B8h] [rbp-98h]
  _BYTE v127[16]; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int8 *v128; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v129; // [rsp+D8h] [rbp-78h]
  unsigned __int64 *v130; // [rsp+E0h] [rbp-70h]

  v10 = (_QWORD *)a1;
  for ( i = sub_157EE30(*(_QWORD *)(a1 + 40)); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v12 = i - 24;
    if ( a1 == i - 24 )
      break;
    if ( *(_BYTE *)(i - 8) == 78 )
    {
      v13 = *(_QWORD *)(i - 48);
      if ( !*(_BYTE *)(v13 + 16) && (*(_BYTE *)(v13 + 33) & 0x20) != 0 && *(_DWORD *)(v13 + 36) == 76 )
      {
        v14 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF));
        if ( *(_BYTE *)(v14 + 16) == 88 )
        {
          v82 = sub_157F120(*(_QWORD *)(v14 + 40));
          v14 = sub_157EBA0(v82);
        }
        v15 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v15 + 16) == 88 )
        {
          v81 = sub_157F120(*(_QWORD *)(v15 + 40));
          v15 = sub_157EBA0(v81);
        }
        if ( v15 == v14 )
        {
          v16 = *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(i - 4) & 0xFFFFFFF)));
          v17 = *(_QWORD **)(v16 + 24);
          if ( *(_DWORD *)(v16 + 32) > 0x40u )
            v17 = (_QWORD *)*v17;
          v18 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          v19 = *(_QWORD **)(v18 + 24);
          if ( *(_DWORD *)(v18 + 32) > 0x40u )
            v19 = (_QWORD *)*v19;
          if ( (_DWORD)v17 == (_DWORD)v19 )
          {
            sub_15F22F0((_QWORD *)a1, i - 24);
            break;
          }
        }
      }
    }
  }
  v20 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v20 )
  {
    v21 = *(__int64 **)a2;
    v114 = 0;
    while ( 1 )
    {
      v22 = *v21;
      v23 = *(_DWORD *)(*v21 + 20) & 0xFFFFFFF;
      v24 = *(_QWORD *)(*v21 + 24 * (1 - v23));
      v25 = *(_DWORD *)(v24 + 32);
      v26 = *(_QWORD **)(v24 + 24);
      if ( v25 > 0x40 )
        v26 = (_QWORD *)*v26;
      v27 = *(_QWORD *)(v22 + 24 * (2 - v23));
      v28 = *(_QWORD **)(v27 + 24);
      if ( *(_DWORD *)(v27 + 32) > 0x40u )
        v28 = (_QWORD *)*v28;
      if ( (_DWORD)v26 == (_DWORD)v28 || v10[5] != *(_QWORD *)(v22 + 40) )
        goto LABEL_22;
      v29 = *(_QWORD *)(v22 - 24 * v23);
      v30 = *(_BYTE *)(v29 + 16);
      v31 = v29;
      v32 = v30;
      if ( v30 == 88 )
      {
        v88 = sub_157F120(*(_QWORD *)(v29 + 40));
        v29 = sub_157EBA0(v88);
        v23 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
        v24 = *(_QWORD *)(v22 + 24 * (1 - v23));
        v31 = *(_QWORD *)(v22 - 24 * v23);
        v30 = *(_BYTE *)(v29 + 16);
        v25 = *(_DWORD *)(v24 + 32);
        v32 = *(_BYTE *)(v31 + 16);
      }
      if ( v30 <= 0x17u )
        break;
      if ( v30 == 78 )
      {
        v85 = v29 | 4;
      }
      else
      {
        v33 = 0;
        if ( v30 != 29 )
          goto LABEL_34;
        v85 = v29 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v33 = v85 & 0xFFFFFFFFFFFFFFF8LL;
      v34 = (v85 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v85 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
      if ( (v85 & 4) == 0 )
        goto LABEL_34;
LABEL_35:
      v35 = *(_QWORD **)(v24 + 24);
      if ( v25 > 0x40 )
        v35 = (_QWORD *)*v35;
      v36 = *(__int64 **)(v34 + 24LL * (unsigned int)v35);
      if ( v32 == 88 )
      {
        v113 = v36;
        v86 = sub_157F120(*(_QWORD *)(v31 + 40));
        v87 = sub_157EBA0(v86);
        v36 = v113;
        v31 = v87;
        v32 = *(_BYTE *)(v87 + 16);
        v23 = *(_DWORD *)(v22 + 20) & 0xFFFFFFF;
      }
      if ( v32 <= 0x17u )
      {
        v37 = 0;
LABEL_42:
        v38 = v37 - 24LL * (*(_DWORD *)(v37 + 20) & 0xFFFFFFF);
        goto LABEL_43;
      }
      if ( v32 == 78 )
      {
        v84 = v31 | 4;
      }
      else
      {
        v37 = 0;
        if ( v32 != 29 )
          goto LABEL_42;
        v84 = v31 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v37 = v84 & 0xFFFFFFFFFFFFFFF8LL;
      v38 = (v84 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v84 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
      if ( (v84 & 4) == 0 )
        goto LABEL_42;
LABEL_43:
      v39 = *(_QWORD *)(v22 + 24 * (2 - v23));
      v40 = *(_QWORD **)(v39 + 24);
      if ( *(_DWORD *)(v39 + 32) > 0x40u )
        v40 = (_QWORD *)*v40;
      v41 = *(_QWORD *)(v38 + 24LL * (unsigned int)v40);
      if ( *(_BYTE *)(v41 + 16) != 56 )
        goto LABEL_22;
      v42 = *(_DWORD *)(v41 + 20) & 0xFFFFFFF;
      if ( *(__int64 **)(v41 - 24 * v42) != v36 )
        goto LABEL_22;
      v125 = (__int64 **)v127;
      v126 = 0x200000000LL;
      if ( (unsigned int)v42 > 1 )
      {
        v43 = v41 - 24 * v42;
        while ( 1 )
        {
          v44 = *(_QWORD *)(v43 + 24);
          if ( *(_BYTE *)(v44 + 16) != 13 )
            goto LABEL_22;
          v45 = *(_DWORD *)(v44 + 32) <= 0x40u ? *(_QWORD *)(v44 + 24) : **(_QWORD **)(v44 + 24);
          if ( v45 > 0x14 )
            goto LABEL_22;
          v43 += 24;
          if ( v41 + 24 * ((unsigned int)(v42 - 2) - v42) + 24 == v43 )
          {
            v46 = 1;
            v115 = v21;
            v47 = 2;
            v48 = v20;
            v107 = v10;
            v49 = v22;
            v50 = 0;
            v51 = 1;
            v52 = v41;
            while ( 1 )
            {
              v53 = *(__int64 **)(v52 + 24 * (v51 - v42));
              if ( (unsigned int)v50 >= v47 )
              {
                v105 = v36;
                v106 = v53;
                sub_16CD150((__int64)&v125, v127, 0, 8, v46, v41);
                v50 = (unsigned int)v126;
                v36 = v105;
                v53 = v106;
              }
              ++v51;
              v125[v50] = v53;
              v50 = (unsigned int)(v126 + 1);
              v54 = *(_DWORD *)(v52 + 20);
              LODWORD(v126) = v126 + 1;
              v42 = v54 & 0xFFFFFFF;
              if ( (unsigned int)v42 <= (unsigned int)v51 )
                break;
              v47 = HIDWORD(v126);
            }
            v22 = v49;
            v41 = v52;
            v20 = v48;
            v10 = v107;
            v21 = v115;
            break;
          }
        }
      }
      v55 = v10[4];
      if ( v55 == v10[5] + 40LL || (v56 = v55 - 24, !v55) )
        v56 = 0;
      v108 = v36;
      v116 = v41;
      sub_17CE510((__int64)&v128, v56, 0, 0, 0);
      v57 = v116;
      v58 = v108;
      v123[0] = *(unsigned __int8 **)(v22 + 48);
      if ( v123[0] )
      {
        sub_1623A60((__int64)v123, (__int64)v123[0], 2);
        v59 = (__int64)v128;
        v57 = v116;
        v58 = v108;
        if ( !v128 )
          goto LABEL_65;
      }
      else
      {
        v59 = (__int64)v128;
        if ( !v128 )
          goto LABEL_67;
      }
      v109 = v58;
      v117 = v57;
      sub_161E7C0((__int64)&v128, v59);
      v58 = v109;
      v57 = v117;
LABEL_65:
      v128 = v123[0];
      if ( v123[0] )
      {
        v110 = v58;
        v118 = v57;
        sub_1623210((__int64)v123, v123[0], (__int64)&v128);
        v58 = v110;
        v123[0] = 0;
        v57 = v118;
      }
LABEL_67:
      v103 = v58;
      v119 = v57;
      sub_17CD270((__int64 *)v123);
      v111 = v10;
      v60 = v119;
      if ( *v103 != *v10 )
      {
        v124 = 257;
        v61 = sub_12AA3B0((__int64 *)&v128, 0x2Fu, (__int64)v10, *v103, (__int64)v123);
        v60 = v119;
        v111 = (_BYTE *)v61;
      }
      v62 = *(_QWORD *)(v60 + 56);
      v122 = 257;
      v104 = v62;
      if ( v111[16] > 0x10u )
        goto LABEL_74;
      if ( (_DWORD)v126 )
      {
        v63 = 0;
        while ( *((_BYTE *)v125[v63] + 16) <= 0x10u )
        {
          if ( (unsigned int)v126 == ++v63 )
            goto LABEL_113;
        }
LABEL_74:
        v124 = 257;
        v102 = v126 + 1;
        if ( !v104 )
        {
          v89 = *(_QWORD *)v111;
          if ( *(_BYTE *)(*(_QWORD *)v111 + 8LL) == 16 )
            v89 = **(_QWORD **)(v89 + 16);
          v104 = *(_QWORD *)(v89 + 24);
        }
        v98 = (unsigned int)v126;
        v100 = v125;
        v64 = sub_1648A60(72, v102);
        v65 = (__int64)v100;
        v66 = v98;
        v120 = v64;
        if ( v64 )
        {
          v101 = (__int64)v64;
          v99 = (__int64)&v64[-3 * v102];
          v67 = *(_QWORD *)v111;
          if ( *(_BYTE *)(*(_QWORD *)v111 + 8LL) == 16 )
            v67 = **(_QWORD **)(v67 + 16);
          v92 = (__int64 *)v65;
          v95 = v66;
          v91 = *(_DWORD *)(v67 + 8) >> 8;
          v68 = (__int64 *)sub_15F9F50(v104, v65, v66);
          v69 = (__int64 *)sub_1646BA0(v68, v91);
          v70 = v95;
          v71 = v92;
          v72 = v69;
          if ( *(_BYTE *)(*(_QWORD *)v111 + 8LL) == 16 )
          {
            v94 = v95;
            v97 = v71;
            v90 = sub_16463B0(v69, *(_QWORD *)(*(_QWORD *)v111 + 32LL));
            v71 = v97;
            v70 = v94;
            v72 = v90;
          }
          else
          {
            v73 = &v92[v95];
            if ( v92 != v73 )
            {
              v74 = v92;
              while ( 1 )
              {
                v75 = *(_QWORD *)*v74;
                if ( *(_BYTE *)(v75 + 8) == 16 )
                  break;
                if ( v73 == ++v74 )
                  goto LABEL_84;
              }
              v76 = sub_16463B0(v72, *(_QWORD *)(v75 + 32));
              v70 = v95;
              v71 = v92;
              v72 = v76;
            }
          }
LABEL_84:
          v93 = v70;
          v96 = v71;
          sub_15F1EA0((__int64)v120, (__int64)v72, 32, v99, v102, 0);
          v120[7] = v104;
          v120[8] = sub_15F9F50(v104, (__int64)v96, v93);
          sub_15F9CE0((__int64)v120, (__int64)v111, v96, v93, (__int64)v123);
        }
        else
        {
          v101 = 0;
        }
        if ( v129 )
        {
          v112 = v130;
          sub_157E9D0(v129 + 40, (__int64)v120);
          v77 = *v112;
          v78 = v120[3];
          v120[4] = v112;
          v77 &= 0xFFFFFFFFFFFFFFF8LL;
          v120[3] = v77 | v78 & 7;
          *(_QWORD *)(v77 + 8) = v120 + 3;
          *v112 = *v112 & 7 | (unsigned __int64)(v120 + 3);
        }
        sub_164B780(v101, &v121);
        sub_12A86E0((__int64 *)&v128, (__int64)v120);
        sub_164B7C0((__int64)v120, v22);
        goto LABEL_88;
      }
LABEL_113:
      BYTE4(v123[0]) = 0;
      v120 = (_QWORD *)sub_15A2E80(v104, (__int64)v111, v125, (unsigned int)v126, 0, (__int64)v123, 0);
      sub_164B7C0((__int64)v120, v22);
LABEL_88:
      if ( *(_QWORD *)v22 != *v120 )
      {
        v124 = 257;
        v120 = (_QWORD *)sub_12AA3B0((__int64 *)&v128, 0x2Fu, (__int64)v120, *(_QWORD *)v22, (__int64)v123);
      }
      sub_164D160(v22, (__int64)v120, a3, a4, a5, a6, v79, v80, a9, a10);
      sub_15F20C0((_QWORD *)v22);
      sub_17CD270((__int64 *)&v128);
      if ( v125 != (__int64 **)v127 )
        _libc_free((unsigned __int64)v125);
      v114 = 1;
LABEL_22:
      if ( (__int64 *)v20 == ++v21 )
        return v114;
    }
    v33 = 0;
LABEL_34:
    v34 = v33 - 24LL * (*(_DWORD *)(v33 + 20) & 0xFFFFFFF);
    goto LABEL_35;
  }
  return 0;
}
