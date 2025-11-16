// Function: sub_2C34E00
// Address: 0x2c34e00
//
__int64 __fastcall sub_2C34E00(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // r15
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // r13
  int v41; // ebx
  unsigned __int64 v42; // rax
  unsigned int v43; // ebx
  int v44; // r9d
  __int64 v45; // r8
  unsigned int v46; // eax
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 v49; // r12
  __int64 v50; // rcx
  _QWORD *v51; // rdi
  _QWORD *v52; // rax
  __int64 *v53; // r11
  __int64 v54; // rcx
  size_t v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // rax
  __int64 v59; // rcx
  int v60; // edx
  __int64 v61; // rax
  __int64 v62; // r9
  __int64 v63; // r14
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  __int64 v73; // r9
  __int64 v74; // rbx
  __int64 v75; // r14
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rcx
  unsigned int v79; // r11d
  char v80; // al
  __int64 v82; // [rsp+38h] [rbp-568h]
  _QWORD **v83; // [rsp+48h] [rbp-558h]
  __int64 v84; // [rsp+50h] [rbp-550h]
  __int64 v85; // [rsp+58h] [rbp-548h]
  __int64 v86; // [rsp+60h] [rbp-540h]
  __int64 v87; // [rsp+68h] [rbp-538h]
  char v89; // [rsp+78h] [rbp-528h]
  int v90; // [rsp+8Ch] [rbp-514h]
  __int64 *v91; // [rsp+90h] [rbp-510h]
  __int64 v92; // [rsp+98h] [rbp-508h]
  __int64 v93; // [rsp+A8h] [rbp-4F8h]
  __int64 v94; // [rsp+B8h] [rbp-4E8h] BYREF
  __int64 v95; // [rsp+C0h] [rbp-4E0h] BYREF
  __int64 *v96; // [rsp+C8h] [rbp-4D8h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-4D0h] BYREF
  __int64 v98; // [rsp+D8h] [rbp-4C8h]
  __int64 v99; // [rsp+E0h] [rbp-4C0h] BYREF
  __int64 v100; // [rsp+E8h] [rbp-4B8h]
  __int64 v101; // [rsp+F0h] [rbp-4B0h]
  unsigned int v102; // [rsp+F8h] [rbp-4A8h]
  __int64 v103; // [rsp+100h] [rbp-4A0h] BYREF
  __int64 v104; // [rsp+108h] [rbp-498h]
  __int64 v105; // [rsp+110h] [rbp-490h]
  __int64 v106; // [rsp+118h] [rbp-488h]
  _QWORD *v107; // [rsp+120h] [rbp-480h]
  _QWORD *v108; // [rsp+128h] [rbp-478h]
  _QWORD v109[12]; // [rsp+130h] [rbp-470h] BYREF
  __int64 v110; // [rsp+190h] [rbp-410h]
  __int64 v111; // [rsp+198h] [rbp-408h]
  __int16 v112; // [rsp+1A8h] [rbp-3F8h]
  _QWORD v113[15]; // [rsp+1B0h] [rbp-3F0h] BYREF
  __int16 v114; // [rsp+228h] [rbp-378h]
  __int16 v115; // [rsp+238h] [rbp-368h]
  _QWORD v116[12]; // [rsp+240h] [rbp-360h] BYREF
  __int64 v117; // [rsp+2A0h] [rbp-300h]
  __int64 v118; // [rsp+2A8h] [rbp-2F8h]
  __int16 v119; // [rsp+2B8h] [rbp-2E8h] BYREF
  _QWORD v120[15]; // [rsp+2C0h] [rbp-2E0h] BYREF
  __int16 v121; // [rsp+338h] [rbp-268h]
  __int16 v122; // [rsp+348h] [rbp-258h]
  _BYTE v123[120]; // [rsp+350h] [rbp-250h] BYREF
  __int16 v124; // [rsp+3C8h] [rbp-1D8h]
  _BYTE v125[120]; // [rsp+3D0h] [rbp-1D0h] BYREF
  __int16 v126; // [rsp+448h] [rbp-158h]
  __int16 v127; // [rsp+458h] [rbp-148h]
  _BYTE v128[120]; // [rsp+460h] [rbp-140h] BYREF
  __int16 v129; // [rsp+4D8h] [rbp-C8h]
  _BYTE v130[120]; // [rsp+4E0h] [rbp-C0h] BYREF
  __int16 v131; // [rsp+558h] [rbp-48h]
  __int16 v132; // [rsp+568h] [rbp-38h]

  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v2 = sub_2AAFF80((__int64)a1);
  if ( !*(_DWORD *)(v2 + 56) )
    BUG();
  v3 = **(_QWORD **)(v2 + 48);
  v103 = 0;
  v4 = *(_QWORD **)(*(_QWORD *)(v3 + 40) + 8LL);
  v104 = 0;
  v105 = 0;
  v83 = (_QWORD **)v4;
  v106 = 0;
  v107 = v4;
  v108 = (_QWORD *)*v4;
  v5 = sub_2BF3F10(a1);
  v82 = v5;
  if ( v5 )
  {
    if ( *(_DWORD *)(v5 + 64) == 1 )
      v82 = **(_QWORD **)(v5 + 56);
    else
      v82 = 0;
  }
  v6 = sub_2BF3F10(a1);
  sub_2C2F4B0(v116, v6);
  sub_2C31060((__int64)v123, (__int64)v116, v7, v8, v9, v10);
  sub_2AB1B50((__int64)&v119);
  sub_2AB1B50((__int64)v116);
  sub_2ABCC20(v109, (__int64)v123, v11, v12, v13, v14);
  v112 = v124;
  sub_2ABCC20(v113, (__int64)v125, v15, v16, v17, v18);
  v114 = v126;
  v115 = v127;
  sub_2ABCC20(v116, (__int64)v128, v19, v20, v21, v22);
  v119 = v129;
  sub_2ABCC20(v120, (__int64)v130, v23, v24, v25, v26);
  v121 = v131;
  v122 = v132;
  while ( 1 )
  {
    v29 = v111;
    v30 = v110;
    v31 = v117;
    if ( v111 - v110 != v118 - v117 )
      goto LABEL_7;
    if ( v110 == v111 )
      break;
    while ( *(_QWORD *)v30 == *(_QWORD *)v31 )
    {
      v80 = *(_BYTE *)(v30 + 24);
      if ( v80 != *(_BYTE *)(v31 + 24)
        || v80 && (*(_QWORD *)(v30 + 8) != *(_QWORD *)(v31 + 8) || *(_QWORD *)(v30 + 16) != *(_QWORD *)(v31 + 16)) )
      {
        break;
      }
      v30 += 32;
      v31 += 32;
      if ( v111 == v30 )
        goto LABEL_94;
    }
LABEL_7:
    v32 = *(_QWORD *)(v111 - 32);
    v33 = (unsigned __int64)&v103;
    v87 = v32 + 112;
    v93 = *(_QWORD *)(v32 + 120);
    if ( v32 + 112 != v93 )
    {
      while ( 1 )
      {
        v34 = v93;
        v33 = *(unsigned __int8 *)(v93 - 16);
        v93 = *(_QWORD *)(v93 + 8);
        if ( (unsigned __int8)v33 <= 0x18u )
        {
          v35 = 26280448;
          if ( _bittest64(&v35, v33) )
          {
            v36 = *(_QWORD *)(v34 - 8) & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)(v34 - 8) & 4) != 0 )
              v36 = **(_QWORD **)v36;
            v27 = *(_QWORD *)(v36 + 40);
            v37 = *(_QWORD *)(a2 + 8);
            v38 = *(unsigned int *)(a2 + 24);
            if ( (_DWORD)v38 )
            {
              v31 = ((_DWORD)v38 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
              v29 = v37 + 16 * v31;
              v28 = *(_QWORD *)v29;
              if ( v27 != *(_QWORD *)v29 )
              {
                v29 = 1;
                while ( v28 != -4096 )
                {
                  v79 = v29 + 1;
                  v31 = ((_DWORD)v38 - 1) & (unsigned int)(v29 + v31);
                  v29 = v37 + 16LL * (unsigned int)v31;
                  v28 = *(_QWORD *)v29;
                  if ( v27 == *(_QWORD *)v29 )
                    goto LABEL_16;
                  v29 = v79;
                }
                goto LABEL_9;
              }
LABEL_16:
              if ( v29 != v37 + 16 * v38 )
              {
                v39 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 16LL * *(unsigned int *)(v29 + 8) + 8);
                v90 = v39;
                LOBYTE(v29) = (_DWORD)v39 == 0;
                if ( (_DWORD)v39 != 0 && (_BYTE)v33 != 16 && (_BYTE)v33 != 9 )
                  break;
              }
            }
          }
        }
LABEL_9:
        if ( v87 == v93 )
          goto LABEL_60;
      }
      v40 = sub_2BFD6A0((__int64)&v103, v36);
      v31 = (unsigned int)v39;
      v41 = sub_BCB060(v40);
      v84 = sub_BCCE00(*v83, v39);
      v42 = *(unsigned __int8 *)(v34 - 16);
      if ( (unsigned __int8)v42 > 0x17u )
      {
        v33 = v34 - 24;
        v85 = v34 - 24;
        if ( (_DWORD)v39 == v41 )
          goto LABEL_20;
        goto LABEL_67;
      }
      v29 = 8860176;
      if ( _bittest64(&v29, v42) )
      {
        v29 = *(unsigned __int8 *)(v34 + 128);
        switch ( *(_BYTE *)(v34 + 128) )
        {
          case 1:
            *(_BYTE *)(v34 + 132) &= 0xFCu;
            break;
          case 2:
          case 3:
          case 6:
            *(_BYTE *)(v34 + 132) &= ~1u;
            break;
          case 4:
            *(_DWORD *)(v34 + 132) = 0;
            break;
          case 5:
            *(_BYTE *)(v34 + 132) &= 0xF9u;
            break;
          default:
            v29 = 8860176;
            break;
        }
      }
      v33 = v34 - 24;
      v85 = v34 - 24;
      if ( (_DWORD)v39 == v41 )
      {
LABEL_70:
        if ( (_BYTE)v42 == 20 )
          goto LABEL_9;
        goto LABEL_20;
      }
      switch ( (_BYTE)v42 )
      {
        case 0x17:
          goto LABEL_66;
        case 9:
          v29 = *(_QWORD *)(v34 + 112);
          if ( *(_BYTE *)v29 != 82 )
            goto LABEL_67;
          break;
        case 0x10:
LABEL_66:
          if ( *(_DWORD *)(v34 + 136) != 53 )
            goto LABEL_67;
          break;
        default:
          if ( (_BYTE)v42 != 4 || *(_BYTE *)(v34 + 136) != 53 )
          {
LABEL_67:
            v72 = sub_22077B0(0xB0u);
            v74 = v72;
            if ( v72 )
            {
              v94 = 0;
              v95 = 0;
              v96 = (__int64 *)v36;
              v97 = 0;
              sub_2AAF310(v72, 16, (__int64 *)&v96, 1, &v97, v73);
              v75 = v74 + 96;
              sub_9C6650(&v97);
              sub_2BF0340(v74 + 96, 1, 0, v74, v76, v77);
              *(_QWORD *)v74 = &unk_4A231C8;
              *(_QWORD *)(v74 + 40) = &unk_4A23200;
              *(_QWORD *)(v74 + 96) = &unk_4A23238;
              sub_9C6650(&v95);
              *(_BYTE *)(v74 + 152) = 7;
              *(_DWORD *)(v74 + 156) = 0;
              *(_QWORD *)v74 = &unk_4A23258;
              *(_QWORD *)(v74 + 40) = &unk_4A23290;
              *(_QWORD *)(v74 + 96) = &unk_4A232C8;
              sub_9C6650(&v94);
              *(_QWORD *)(v74 + 168) = v40;
              *(_DWORD *)(v74 + 160) = 39;
              *(_QWORD *)v74 = &unk_4A23F58;
              *(_QWORD *)(v74 + 40) = &unk_4A23F90;
              *(_QWORD *)(v74 + 96) = &unk_4A23FC8;
              sub_2C19DE0((_QWORD *)v74, v85);
            }
            else
            {
              v75 = 0;
              sub_2C19DE0(0, v85);
            }
            sub_2BF1250(v36, v75);
            v31 = 0;
            sub_2AAED30(v74 + 40, 0, v36);
            LOBYTE(v42) = *(_BYTE *)(v34 - 16);
            goto LABEL_70;
          }
          break;
      }
LABEL_20:
      if ( ((_BYTE)v42 == 24) != *(_DWORD *)(v34 + 32) )
      {
        v43 = (_BYTE)v42 == 24;
        while ( 1 )
        {
          v57 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8LL * v43);
          v31 = v57;
          v58 = sub_2BFD6A0((__int64)&v103, v57);
          if ( v90 != (unsigned int)sub_BCB060(v58) )
            break;
LABEL_34:
          if ( ++v43 == *(_DWORD *)(v34 + 32) )
            goto LABEL_9;
        }
        v97 = v57;
        v98 = 0;
        if ( !v102 )
        {
          ++v99;
          v96 = 0;
          goto LABEL_38;
        }
        v44 = 1;
        v45 = 0;
        v46 = (v102 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v91 = (__int64 *)(v100 + 16LL * v46);
        v47 = *v91;
        if ( v57 == *v91 )
        {
LABEL_23:
          v89 = 0;
          v48 = v91[1];
          v92 = v34 + 16;
          if ( v48 )
          {
            v49 = v91[1];
            v48 += 96;
          }
          else
          {
            v49 = 0;
          }
LABEL_25:
          v50 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8LL * v43);
          v97 = v92;
          v51 = *(_QWORD **)(v50 + 16);
          v86 = v50;
          v31 = (__int64)&v51[*(unsigned int *)(v50 + 24)];
          v52 = sub_2C25810(v51, v31, &v97);
          if ( (_QWORD *)v31 != v52 )
          {
            v54 = v86;
            if ( (_QWORD *)v31 != v52 + 1 )
            {
              v55 = v31 - (_QWORD)(v52 + 1);
              v31 = (__int64)(v52 + 1);
              memmove(v52, v52 + 1, v55);
              v54 = v86;
              LODWORD(v28) = *(_DWORD *)(v86 + 24);
            }
            v28 = (unsigned int)(v28 - 1);
            *(_DWORD *)(v54 + 24) = v28;
            v53 = (__int64 *)(*(_QWORD *)(v34 + 24) + 8LL * v43);
          }
          *v53 = v48;
          v56 = *(unsigned int *)(v48 + 24);
          if ( v56 + 1 > (unsigned __int64)*(unsigned int *)(v48 + 28) )
          {
            v31 = v48 + 32;
            sub_C8D5F0(v48 + 16, (const void *)(v48 + 32), v56 + 1, 8u, v27, v28);
            v56 = *(unsigned int *)(v48 + 24);
          }
          v29 = *(_QWORD *)(v48 + 16);
          v33 = v92;
          *(_QWORD *)(v29 + 8 * v56) = v92;
          ++*(_DWORD *)(v48 + 24);
          if ( v89 )
          {
            v91[1] = v49;
            if ( sub_2BF04A0(v57) )
            {
              v31 = v85;
              sub_2C19D60((_QWORD *)v49, v85);
            }
            else
            {
              v31 = v82;
              *(_QWORD *)(v49 + 80) = v82;
              v78 = *(_QWORD *)(v82 + 112);
              *(_QWORD *)(v49 + 32) = v82 + 112;
              v33 = v78 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v49 + 24) = v33 | *(_QWORD *)(v49 + 24) & 7LL;
              *(_QWORD *)(v33 + 8) = v49 + 24;
              v29 = *(_QWORD *)(v82 + 112) & 7LL | (v49 + 24);
              *(_QWORD *)(v82 + 112) = v29;
            }
          }
          goto LABEL_34;
        }
        while ( v47 != -4096 )
        {
          if ( !v45 && v47 == -8192 )
            v45 = (__int64)v91;
          v46 = (v102 - 1) & (v44 + v46);
          v91 = (__int64 *)(v100 + 16LL * v46);
          v47 = *v91;
          if ( v57 == *v91 )
            goto LABEL_23;
          ++v44;
        }
        if ( !v45 )
          v45 = (__int64)v91;
        ++v99;
        v60 = v101 + 1;
        v91 = (__int64 *)v45;
        v96 = (__int64 *)v45;
        if ( 4 * ((int)v101 + 1) < 3 * v102 )
        {
          v59 = v57;
          if ( v102 - HIDWORD(v101) - v60 <= v102 >> 3 )
          {
            sub_2C34C20((__int64)&v99, v102);
LABEL_39:
            sub_2C2BEC0((__int64)&v99, &v97, &v96);
            v59 = v97;
            v60 = v101 + 1;
            v91 = v96;
          }
          LODWORD(v101) = v60;
          if ( *v91 != -4096 )
            --HIDWORD(v101);
          *v91 = v59;
          v91[1] = v98;
          v61 = sub_22077B0(0xB0u);
          v49 = v61;
          v92 = v34 + 16;
          if ( v61 )
          {
            *(_QWORD *)(v61 + 64) = v57;
            v63 = v61 + 40;
            v95 = 0;
            *(_BYTE *)(v61 + 8) = 16;
            *(_QWORD *)v61 = &unk_4A231A8;
            v96 = 0;
            v97 = 0;
            *(_QWORD *)(v61 + 40) = &unk_4A23170;
            *(_QWORD *)(v61 + 48) = v61 + 64;
            *(_QWORD *)(v61 + 24) = 0;
            *(_QWORD *)(v61 + 32) = 0;
            *(_QWORD *)(v61 + 16) = 0;
            *(_QWORD *)(v61 + 56) = 0x200000001LL;
            v64 = *(unsigned int *)(v57 + 24);
            if ( v64 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 28) )
            {
              sub_C8D5F0(v57 + 16, (const void *)(v57 + 32), v64 + 1, 8u, v64 + 1, v62);
              v64 = *(unsigned int *)(v57 + 24);
            }
            *(_QWORD *)(*(_QWORD *)(v57 + 16) + 8 * v64) = v63;
            ++*(_DWORD *)(v57 + 24);
            *(_QWORD *)(v49 + 80) = 0;
            *(_QWORD *)(v49 + 40) = &unk_4A23AA8;
            v65 = v97;
            *(_QWORD *)v49 = &unk_4A23A70;
            *(_QWORD *)(v49 + 88) = v65;
            if ( v65 )
              sub_2C25AB0((__int64 *)(v49 + 88));
            v48 = v49 + 96;
            sub_9C6650(&v97);
            sub_2BF0340(v49 + 96, 1, 0, v49, v66, v67);
            *(_QWORD *)v49 = &unk_4A231C8;
            *(_QWORD *)(v49 + 40) = &unk_4A23200;
            *(_QWORD *)(v49 + 96) = &unk_4A23238;
            sub_9C6650(&v96);
            *(_BYTE *)(v49 + 152) = 7;
            *(_DWORD *)(v49 + 156) = 0;
            *(_QWORD *)v49 = &unk_4A23258;
            *(_QWORD *)(v49 + 40) = &unk_4A23290;
            *(_QWORD *)(v49 + 96) = &unk_4A232C8;
            sub_9C6650(&v95);
            *(_DWORD *)(v49 + 160) = 38;
            v89 = 1;
            *(_QWORD *)v49 = &unk_4A23F58;
            *(_QWORD *)(v49 + 96) = &unk_4A23FC8;
            *(_QWORD *)(v49 + 40) = &unk_4A23F90;
            *(_QWORD *)(v49 + 168) = v84;
          }
          else
          {
            v89 = 1;
            v48 = 0;
          }
          goto LABEL_25;
        }
LABEL_38:
        sub_2C34C20((__int64)&v99, 2 * v102);
        goto LABEL_39;
      }
      goto LABEL_9;
    }
LABEL_60:
    sub_2AD7320((__int64)v109, v31, v29, v33, v27, v28);
    sub_2C30FC0(v109, v31, v68, v69, v70, v71);
  }
LABEL_94:
  sub_2AB1B50((__int64)v120);
  sub_2AB1B50((__int64)v116);
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v109);
  sub_2AB1B50((__int64)v130);
  sub_2AB1B50((__int64)v128);
  sub_2AB1B50((__int64)v125);
  sub_2AB1B50((__int64)v123);
  sub_C7D6A0(v104, 16LL * (unsigned int)v106, 8);
  return sub_C7D6A0(v100, 16LL * v102, 8);
}
