// Function: sub_194C320
// Address: 0x194c320
//
_QWORD *__fastcall sub_194C320(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // rdi
  __int64 *v4; // r15
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // r12
  unsigned int v11; // esi
  int v12; // r11d
  __int64 v13; // rax
  _QWORD *v14; // rbx
  int v15; // edx
  __int64 v16; // rdx
  unsigned __int64 *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // edx
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rbx
  __int64 v35; // rax
  char v36; // di
  _QWORD *result; // rax
  __int64 v38; // rbx
  __int64 v39; // r11
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // r15
  __int64 v43; // rbx
  __int64 v44; // r12
  __int64 v45; // r12
  unsigned int v46; // r15d
  _QWORD *v47; // r14
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rbx
  _QWORD *v51; // rdx
  _QWORD *v52; // rdx
  _QWORD *v53; // r14
  __int64 v54; // rbx
  __int64 v55; // rcx
  __int64 v56; // r8
  unsigned int v57; // eax
  __int64 v58; // r9
  int v59; // edx
  __int64 v60; // rsi
  __int64 v61; // rcx
  __int64 v62; // rdi
  __int64 v63; // rsi
  __int64 v64; // r13
  __int64 v65; // rsi
  __int64 v66; // rcx
  __int64 v67; // rdi
  __int64 v68; // r10
  __int64 v69; // rdx
  unsigned int v70; // ecx
  int v71; // eax
  __int64 v72; // rdx
  _QWORD *v73; // rax
  __int64 v74; // rcx
  unsigned __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // rcx
  _QWORD *v81; // rdx
  int v82; // r10d
  _QWORD *v83; // r9
  int v84; // edi
  int v85; // r11d
  int v86; // r11d
  __int64 v87; // r8
  int v88; // esi
  _QWORD *v89; // rcx
  __int64 v90; // rdx
  __int64 v91; // rdi
  int v92; // r11d
  __int64 v93; // r8
  int v94; // esi
  __int64 v95; // rdx
  __int64 v96; // rdi
  unsigned int v97; // [rsp+4h] [rbp-BCh]
  __int64 v98; // [rsp+10h] [rbp-B0h]
  __int64 v100; // [rsp+20h] [rbp-A0h]
  __int64 v101; // [rsp+28h] [rbp-98h]
  unsigned __int64 v102; // [rsp+28h] [rbp-98h]
  __int64 *v103; // [rsp+30h] [rbp-90h]
  __int64 v104; // [rsp+30h] [rbp-90h]
  unsigned int v105; // [rsp+30h] [rbp-90h]
  int v106; // [rsp+38h] [rbp-88h]
  int v107; // [rsp+38h] [rbp-88h]
  __int64 v109; // [rsp+40h] [rbp-80h]
  __int64 v110; // [rsp+40h] [rbp-80h]
  __int64 v112; // [rsp+48h] [rbp-78h]
  __int64 v113; // [rsp+58h] [rbp-68h] BYREF
  char *v114; // [rsp+60h] [rbp-60h] BYREF
  __int64 v115; // [rsp+68h] [rbp-58h] BYREF
  __int64 v116; // [rsp+70h] [rbp-50h]
  __int64 v117; // [rsp+78h] [rbp-48h]
  __int64 v118; // [rsp+80h] [rbp-40h]

  v3 = a1[7];
  v103 = *(__int64 **)(v3 + 40);
  if ( v103 != *(__int64 **)(v3 + 32) )
  {
    v4 = *(__int64 **)(v3 + 32);
    v5 = a2 + 24;
    while ( 1 )
    {
      v6 = *v4;
      v7 = *a1;
      if ( *a3 )
      {
        v115 = (__int64)a3;
        v114 = ".";
        LOWORD(v116) = 771;
      }
      else
      {
        v114 = ".";
        LOWORD(v116) = 259;
      }
      v8 = sub_1AB5760(v6, v5, &v114, v7, 0, 0);
      v9 = *(_BYTE **)(a2 + 8);
      v113 = v8;
      v10 = v8;
      if ( v9 == *(_BYTE **)(a2 + 16) )
      {
        sub_1292090(a2, v9, &v113);
        v10 = v113;
      }
      else
      {
        if ( v9 )
        {
          *(_QWORD *)v9 = v8;
          v9 = *(_BYTE **)(a2 + 8);
          v10 = v113;
        }
        *(_QWORD *)(a2 + 8) = v9 + 8;
      }
      v115 = 2;
      v116 = 0;
      v117 = v6;
      if ( v6 != -8 && v6 != 0 && v6 != -16 )
        sub_164C220((__int64)&v115);
      v11 = *(_DWORD *)(a2 + 48);
      v118 = v5;
      v114 = (char *)&unk_49E6B50;
      if ( !v11 )
        break;
      v13 = v117;
      v19 = *(_QWORD *)(a2 + 32);
      v20 = (v11 - 1) & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
      v14 = (_QWORD *)(v19 + ((unsigned __int64)v20 << 6));
      v21 = v14[3];
      if ( v117 != v21 )
      {
        v82 = 1;
        v83 = 0;
        while ( v21 != -8 )
        {
          if ( !v83 && v21 == -16 )
            v83 = v14;
          v20 = (v11 - 1) & (v82 + v20);
          v14 = (_QWORD *)(v19 + ((unsigned __int64)v20 << 6));
          v21 = v14[3];
          if ( v117 == v21 )
            goto LABEL_29;
          ++v82;
        }
        v84 = *(_DWORD *)(a2 + 40);
        if ( v83 )
          v14 = v83;
        ++*(_QWORD *)(a2 + 24);
        v15 = v84 + 1;
        if ( 4 * (v84 + 1) < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a2 + 44) - v15 > v11 >> 3 )
            goto LABEL_17;
          sub_12E48B0(v5, v11);
          v85 = *(_DWORD *)(a2 + 48);
          if ( v85 )
          {
            v13 = v117;
            v86 = v85 - 1;
            v87 = *(_QWORD *)(a2 + 32);
            v88 = 1;
            v89 = 0;
            LODWORD(v90) = v86 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
            v14 = (_QWORD *)(v87 + ((unsigned __int64)(unsigned int)v90 << 6));
            v91 = v14[3];
            if ( v117 != v91 )
            {
              while ( v91 != -8 )
              {
                if ( !v89 && v91 == -16 )
                  v89 = v14;
                v90 = v86 & (unsigned int)(v90 + v88);
                v14 = (_QWORD *)(v87 + (v90 << 6));
                v91 = v14[3];
                if ( v117 == v91 )
                  goto LABEL_16;
                ++v88;
              }
LABEL_127:
              if ( v89 )
                v14 = v89;
            }
LABEL_16:
            v15 = *(_DWORD *)(a2 + 40) + 1;
LABEL_17:
            *(_DWORD *)(a2 + 40) = v15;
            if ( v14[3] == -8 )
            {
              v17 = v14 + 1;
              if ( v13 != -8 )
                goto LABEL_22;
            }
            else
            {
              --*(_DWORD *)(a2 + 44);
              v16 = v14[3];
              if ( v16 != v13 )
              {
                v17 = v14 + 1;
                if ( v16 != 0 && v16 != -8 && v16 != -16 )
                {
                  sub_1649B30(v17);
                  v13 = v117;
                }
LABEL_22:
                v14[3] = v13;
                if ( v13 != 0 && v13 != -8 && v13 != -16 )
                  sub_1649AC0(v17, v115 & 0xFFFFFFFFFFFFFFF8LL);
                v13 = v117;
              }
            }
            v18 = v118;
            v14[5] = 6;
            v14[6] = 0;
            v14[4] = v18;
            v14[7] = 0;
            goto LABEL_29;
          }
LABEL_15:
          v13 = v117;
          v14 = 0;
          goto LABEL_16;
        }
LABEL_14:
        sub_12E48B0(v5, 2 * v11);
        v12 = *(_DWORD *)(a2 + 48);
        if ( v12 )
        {
          v13 = v117;
          v92 = v12 - 1;
          v93 = *(_QWORD *)(a2 + 32);
          v94 = 1;
          v89 = 0;
          LODWORD(v95) = v92 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
          v14 = (_QWORD *)(v93 + ((unsigned __int64)(unsigned int)v95 << 6));
          v96 = v14[3];
          if ( v96 != v117 )
          {
            while ( v96 != -8 )
            {
              if ( v96 == -16 && !v89 )
                v89 = v14;
              v95 = v92 & (unsigned int)(v95 + v94);
              v14 = (_QWORD *)(v93 + (v95 << 6));
              v96 = v14[3];
              if ( v117 == v96 )
                goto LABEL_16;
              ++v94;
            }
            goto LABEL_127;
          }
          goto LABEL_16;
        }
        goto LABEL_15;
      }
LABEL_29:
      v114 = (char *)&unk_49EE2B0;
      if ( v13 != -8 && v13 != 0 && v13 != -16 )
        sub_1649B30(&v115);
      v22 = v14[7];
      v23 = v14 + 5;
      if ( v22 != v10 )
      {
        if ( v22 != 0 && v22 != -8 && v22 != -16 )
        {
          sub_1649B30(v23);
          v23 = v14 + 5;
        }
        v14[7] = v10;
        if ( v10 != 0 && v10 != -8 && v10 != -16 )
          sub_164C220((__int64)v23);
      }
      if ( v103 == ++v4 )
      {
        v3 = a1[7];
        goto LABEL_41;
      }
    }
    ++*(_QWORD *)(a2 + 24);
    goto LABEL_14;
  }
LABEL_41:
  v113 = a2;
  v24 = sub_13FCB50(v3);
  v25 = sub_19498B0((__int64)&v113, v24);
  v26 = sub_157EBA0(v25);
  v27 = sub_1627350((__int64 *)a1[1], 0, 0, 0, 1);
  sub_1626100(v26, "irce.loop.clone", 0xFu, v27);
  v28 = a1[14];
  v114 = (char *)v113;
  v101 = sub_19498B0((__int64)&v114, v28);
  v104 = sub_19498B0((__int64)&v114, a1[15]);
  v29 = sub_19498B0((__int64)&v114, a1[16]);
  v30 = sub_19498B0((__int64)&v114, a1[17]);
  v106 = *((_DWORD *)a1 + 36);
  v31 = sub_19498B0((__int64)&v114, a1[19]);
  v32 = sub_19498B0((__int64)&v114, a1[20]);
  v112 = v33;
  v34 = v32;
  sub_19498B0((__int64)&v114, *(_QWORD *)(v33 + 168));
  v35 = sub_19498B0((__int64)&v114, *(_QWORD *)(v112 + 176));
  LOBYTE(v28) = *(_BYTE *)(v112 + 184);
  v36 = *(_BYTE *)(v112 + 185);
  *(_QWORD *)(a2 + 128) = v29;
  *(_QWORD *)(a2 + 112) = v101;
  *(_QWORD *)(a2 + 120) = v104;
  *(_QWORD *)(a2 + 136) = v30;
  *(_QWORD *)(a2 + 152) = v31;
  *(_DWORD *)(a2 + 144) = v106;
  *(_QWORD *)(a2 + 160) = v34;
  *(_QWORD *)(a2 + 176) = v35;
  result = *(_QWORD **)a2;
  *(_QWORD *)(a2 + 104) = a3;
  v38 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 168) = v39;
  *(_BYTE *)(a2 + 184) = v28;
  *(_BYTE *)(a2 + 185) = v36;
  v40 = (v38 - (__int64)result) >> 3;
  if ( (_DWORD)v40 )
  {
    v102 = 0;
    v98 = 8LL * (unsigned int)(v40 - 1);
    while ( 1 )
    {
      v41 = result[v102 / 8];
      v42 = *(_QWORD *)(v41 + 48);
      v100 = v41;
      v43 = v41 + 40;
      v44 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v112 + 56) + 32LL) + v102);
      if ( v43 != v42 )
      {
        v109 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v112 + 56) + 32LL) + v102);
        do
        {
          v45 = 0;
          if ( v42 )
            v45 = v42 - 24;
          sub_1B75040(&v114, a2 + 24, 3, 0, 0);
          sub_1B79630(&v114, v45);
          sub_1B75110(&v114);
          v42 = *(_QWORD *)(v42 + 8);
        }
        while ( v43 != v42 );
        v44 = v109;
      }
      result = (_QWORD *)sub_157EBA0(v44);
      if ( result )
      {
        v107 = sub_15F4D60((__int64)result);
        result = (_QWORD *)sub_157EBA0(v44);
        v110 = (__int64)result;
        if ( v107 )
          break;
      }
LABEL_100:
      if ( v98 == v102 )
        return result;
      v102 += 8LL;
      result = *(_QWORD **)a2;
    }
    v46 = 0;
    while ( 1 )
    {
      v49 = sub_15F4DF0(v110, v46);
      v50 = *(_QWORD *)(v112 + 56);
      v51 = *(_QWORD **)(v50 + 72);
      result = *(_QWORD **)(v50 + 64);
      if ( v51 == result )
      {
        v47 = &result[*(unsigned int *)(v50 + 84)];
        if ( result == v47 )
        {
          v81 = *(_QWORD **)(v50 + 64);
        }
        else
        {
          do
          {
            if ( v49 == *result )
              break;
            ++result;
          }
          while ( v47 != result );
          v81 = v47;
        }
LABEL_65:
        while ( v81 != result )
        {
          if ( *result < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_55;
          ++result;
        }
        if ( result != v47 )
          goto LABEL_56;
      }
      else
      {
        v47 = &v51[*(unsigned int *)(v50 + 80)];
        result = sub_16CC9F0(v50 + 56, v49);
        if ( v49 == *result )
        {
          v79 = *(_QWORD *)(v50 + 72);
          if ( v79 == *(_QWORD *)(v50 + 64) )
            v80 = *(unsigned int *)(v50 + 84);
          else
            v80 = *(unsigned int *)(v50 + 80);
          v81 = (_QWORD *)(v79 + 8 * v80);
          goto LABEL_65;
        }
        v48 = *(_QWORD *)(v50 + 72);
        if ( v48 == *(_QWORD *)(v50 + 64) )
        {
          result = (_QWORD *)(v48 + 8LL * *(unsigned int *)(v50 + 84));
          v81 = result;
          goto LABEL_65;
        }
        result = (_QWORD *)(v48 + 8LL * *(unsigned int *)(v50 + 80));
LABEL_55:
        if ( result != v47 )
          goto LABEL_56;
      }
      result = (_QWORD *)sub_157F280(v49);
      v53 = v52;
      v54 = (__int64)result;
      if ( v52 == result )
      {
LABEL_56:
        if ( ++v46 == v107 )
          goto LABEL_100;
      }
      else
      {
        v105 = v46;
        do
        {
          v55 = 0x17FFFFFFE8LL;
          v56 = *(unsigned int *)(v54 + 56);
          v57 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
          v58 = *(_BYTE *)(v54 + 23) & 0x40;
          v59 = v57;
          if ( v57 )
          {
            v60 = 24LL * (unsigned int)v56 + 8;
            v61 = 0;
            do
            {
              v62 = v54 - 24LL * v57;
              if ( (_BYTE)v58 )
                v62 = *(_QWORD *)(v54 - 8);
              if ( v44 == *(_QWORD *)(v62 + v60) )
              {
                v55 = 24 * v61;
                goto LABEL_76;
              }
              ++v61;
              v60 += 8;
            }
            while ( v57 != (_DWORD)v61 );
            v55 = 0x17FFFFFFE8LL;
          }
LABEL_76:
          if ( (_BYTE)v58 )
            v63 = *(_QWORD *)(v54 - 8);
          else
            v63 = v54 - 24LL * v57;
          v64 = *(_QWORD *)(v63 + v55);
          v65 = v113;
          v66 = *(unsigned int *)(v113 + 48);
          if ( (_DWORD)v66 )
          {
            v67 = *(_QWORD *)(v113 + 32);
            v58 = ((_DWORD)v66 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
            v65 = v67 + (v58 << 6);
            v68 = *(_QWORD *)(v65 + 24);
            if ( v68 == v64 )
            {
LABEL_80:
              v66 = v67 + (v66 << 6);
              if ( v65 != v66 )
                v64 = *(_QWORD *)(v65 + 56);
            }
            else
            {
              v65 = 1;
              while ( v68 != -8 )
              {
                v58 = ((_DWORD)v66 - 1) & (unsigned int)(v65 + v58);
                v97 = v65 + 1;
                v65 = v67 + ((unsigned __int64)(unsigned int)v58 << 6);
                v68 = *(_QWORD *)(v65 + 24);
                if ( v64 == v68 )
                  goto LABEL_80;
                v65 = v97;
              }
            }
          }
          if ( (_DWORD)v56 == v57 )
          {
            sub_15F55D0(v54, v65, v57, v66, v56, v58);
            v59 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
          }
          v69 = (v59 + 1) & 0xFFFFFFF;
          v70 = v69 - 1;
          v71 = v69 | *(_DWORD *)(v54 + 20) & 0xF0000000;
          *(_DWORD *)(v54 + 20) = v71;
          if ( (v71 & 0x40000000) != 0 )
            v72 = *(_QWORD *)(v54 - 8);
          else
            v72 = v54 - 24 * v69;
          v73 = (_QWORD *)(v72 + 24LL * v70);
          if ( *v73 )
          {
            v74 = v73[1];
            v75 = v73[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v75 = v74;
            if ( v74 )
              *(_QWORD *)(v74 + 16) = *(_QWORD *)(v74 + 16) & 3LL | v75;
          }
          *v73 = v64;
          if ( v64 )
          {
            v76 = *(_QWORD *)(v64 + 8);
            v73[1] = v76;
            if ( v76 )
              *(_QWORD *)(v76 + 16) = (unsigned __int64)(v73 + 1) | *(_QWORD *)(v76 + 16) & 3LL;
            v73[2] = (v64 + 8) | v73[2] & 3LL;
            *(_QWORD *)(v64 + 8) = v73;
          }
          v77 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v54 + 23) & 0x40) != 0 )
            v78 = *(_QWORD *)(v54 - 8);
          else
            v78 = v54 - 24 * v77;
          *(_QWORD *)(v78 + 8LL * (unsigned int)(v77 - 1) + 24LL * *(unsigned int *)(v54 + 56) + 8) = v100;
          result = *(_QWORD **)(v54 + 32);
          if ( !result )
            BUG();
          v54 = 0;
          if ( *((_BYTE *)result - 8) == 77 )
            v54 = (__int64)(result - 3);
        }
        while ( v53 != (_QWORD *)v54 );
        ++v46;
        if ( v105 + 1 == v107 )
          goto LABEL_100;
      }
    }
  }
  return result;
}
