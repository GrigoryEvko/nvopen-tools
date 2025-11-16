// Function: sub_20129E0
// Address: 0x20129e0
//
__int64 __fastcall sub_20129E0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // r15d
  int v7; // r13d
  int v8; // r12d
  char v9; // r9
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // edx
  _DWORD *v13; // rax
  int v14; // r10d
  __int64 v15; // r8
  int v16; // edi
  int v17; // r9d
  unsigned int i; // esi
  __int64 v19; // rdx
  unsigned int v20; // esi
  int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // r8
  int v24; // esi
  unsigned int v25; // eax
  int *v26; // rdx
  int v27; // edi
  unsigned int v28; // eax
  __int64 v29; // r8
  int v30; // esi
  unsigned int v31; // eax
  int *v32; // rdx
  int v33; // edi
  unsigned int v34; // eax
  __int64 v35; // r8
  int v36; // esi
  unsigned int v37; // eax
  int *v38; // rdx
  int v39; // edi
  unsigned int v40; // eax
  __int64 v41; // r8
  int v42; // esi
  unsigned int v43; // eax
  int *v44; // rdx
  int v45; // edi
  unsigned int v46; // eax
  __int64 v47; // r8
  int v48; // esi
  unsigned int v49; // eax
  int *v50; // rdx
  int v51; // edi
  unsigned int v52; // eax
  __int64 v53; // r8
  int v54; // esi
  unsigned int v55; // eax
  int *v56; // rdx
  int v57; // edi
  unsigned int v58; // eax
  __int64 v59; // r8
  int v60; // esi
  unsigned int v61; // eax
  int *v62; // rdx
  int v63; // edi
  unsigned int v64; // eax
  __int64 v65; // r8
  int v66; // esi
  unsigned int v67; // eax
  int *v68; // rdx
  int v69; // edi
  unsigned int v70; // eax
  __int64 v71; // rdi
  int v72; // esi
  int *v73; // rdx
  int v74; // r8d
  unsigned int v75; // eax
  int v76; // eax
  int v77; // eax
  int v78; // eax
  int v79; // eax
  int v80; // eax
  int v81; // eax
  int v82; // eax
  int v83; // eax
  unsigned int v84; // esi
  unsigned int v85; // edx
  int v86; // edi
  unsigned int v87; // r11d
  int v88; // edx
  int v89; // ecx
  int v90; // edx
  int v91; // ecx
  int v92; // edx
  int v93; // ecx
  int v94; // edx
  int v95; // ecx
  int v96; // edx
  int v97; // ecx
  int v98; // edx
  int v99; // ecx
  int v100; // edx
  int v101; // ecx
  int v102; // edx
  int v103; // ecx
  int v104; // edx
  int v105; // ecx
  int v106; // r8d
  _DWORD *v107; // rcx
  __int64 v108; // rdi
  int v109; // edx
  unsigned int v110; // esi
  int v111; // r9d
  __int64 v112; // rdi
  int v113; // edx
  unsigned int v114; // esi
  int v115; // r9d
  int v116; // ecx
  _DWORD *v117; // r11
  int v118; // edx
  int v119; // edx
  int v120; // ecx
  int v122; // [rsp+8h] [rbp-38h]
  int v123; // [rsp+Ch] [rbp-34h]

  result = *(unsigned int *)(a2 + 60);
  v123 = result;
  if ( !(_DWORD)result )
    return result;
  v6 = 0;
  v122 = (a2 >> 9) ^ (a2 >> 4);
  do
  {
    v7 = sub_200F8F0(a1, a3, v6);
    v8 = sub_200F8F0(a1, a2, v6);
    if ( v8 != v7 )
    {
      v9 = *(_BYTE *)(a1 + 1296) & 1;
      if ( v9 )
      {
        v10 = a1 + 1304;
        v11 = 7;
      }
      else
      {
        v84 = *(_DWORD *)(a1 + 1312);
        v10 = *(_QWORD *)(a1 + 1304);
        if ( !v84 )
        {
          v85 = *(_DWORD *)(a1 + 1296);
          ++*(_QWORD *)(a1 + 1288);
          v13 = 0;
          v86 = (v85 >> 1) + 1;
          goto LABEL_77;
        }
        v11 = v84 - 1;
      }
      v12 = v11 & (37 * v8);
      v13 = (_DWORD *)(v10 + 8LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
      {
LABEL_7:
        v13[1] = v7;
        goto LABEL_8;
      }
      v106 = 1;
      v107 = 0;
      while ( v14 != -1 )
      {
        if ( v14 == -2 && !v107 )
          v107 = v13;
        v12 = v11 & (v106 + v12);
        v13 = (_DWORD *)(v10 + 8LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_7;
        ++v106;
      }
      v85 = *(_DWORD *)(a1 + 1296);
      v87 = 24;
      v84 = 8;
      if ( v107 )
        v13 = v107;
      ++*(_QWORD *)(a1 + 1288);
      v86 = (v85 >> 1) + 1;
      if ( v9 )
      {
LABEL_78:
        if ( 4 * v86 >= v87 )
        {
          sub_20108A0(a1 + 1288, 2 * v84);
          if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
          {
            v108 = a1 + 1304;
            v109 = 7;
          }
          else
          {
            v118 = *(_DWORD *)(a1 + 1312);
            v108 = *(_QWORD *)(a1 + 1304);
            if ( !v118 )
              goto LABEL_159;
            v109 = v118 - 1;
          }
          v110 = v109 & (37 * v8);
          v13 = (_DWORD *)(v108 + 8LL * v110);
          v111 = *v13;
          if ( v8 == *v13 )
            goto LABEL_128;
          v120 = 1;
          v117 = 0;
          while ( v111 != -1 )
          {
            if ( !v117 && v111 == -2 )
              v117 = v13;
            v110 = v109 & (v120 + v110);
            v13 = (_DWORD *)(v108 + 8LL * v110);
            v111 = *v13;
            if ( v8 == *v13 )
              goto LABEL_128;
            ++v120;
          }
        }
        else
        {
          if ( v84 - *(_DWORD *)(a1 + 1300) - v86 > v84 >> 3 )
          {
LABEL_80:
            *(_DWORD *)(a1 + 1296) = (2 * (v85 >> 1) + 2) | v85 & 1;
            if ( *v13 != -1 )
              --*(_DWORD *)(a1 + 1300);
            *v13 = v8;
            v13[1] = 0;
            goto LABEL_7;
          }
          sub_20108A0(a1 + 1288, v84);
          if ( (*(_BYTE *)(a1 + 1296) & 1) != 0 )
          {
            v112 = a1 + 1304;
            v113 = 7;
          }
          else
          {
            v119 = *(_DWORD *)(a1 + 1312);
            v112 = *(_QWORD *)(a1 + 1304);
            if ( !v119 )
            {
LABEL_159:
              *(_DWORD *)(a1 + 1296) = (2 * (*(_DWORD *)(a1 + 1296) >> 1) + 2) | *(_DWORD *)(a1 + 1296) & 1;
              BUG();
            }
            v113 = v119 - 1;
          }
          v114 = v113 & (37 * v8);
          v13 = (_DWORD *)(v112 + 8LL * v114);
          v115 = *v13;
          if ( v8 == *v13 )
          {
LABEL_128:
            v85 = *(_DWORD *)(a1 + 1296);
            goto LABEL_80;
          }
          v116 = 1;
          v117 = 0;
          while ( v115 != -1 )
          {
            if ( !v117 && v115 == -2 )
              v117 = v13;
            v114 = v113 & (v116 + v114);
            v13 = (_DWORD *)(v112 + 8LL * v114);
            v115 = *v13;
            if ( v8 == *v13 )
              goto LABEL_128;
            ++v116;
          }
        }
        if ( v117 )
          v13 = v117;
        goto LABEL_128;
      }
      v84 = *(_DWORD *)(a1 + 1312);
LABEL_77:
      v87 = 3 * v84;
      goto LABEL_78;
    }
LABEL_8:
    if ( (*(_BYTE *)(a1 + 144) & 1) != 0 )
    {
      v15 = a1 + 152;
      v16 = 7;
    }
    else
    {
      v21 = *(_DWORD *)(a1 + 160);
      v15 = *(_QWORD *)(a1 + 152);
      if ( !v21 )
        goto LABEL_18;
      v16 = v21 - 1;
    }
    v17 = 1;
    for ( i = v16 & (v122 + v6); ; i = v16 & v20 )
    {
      v19 = v15 + 24LL * i;
      if ( a2 == *(_QWORD *)v19 )
        break;
      if ( !*(_QWORD *)v19 && *(_DWORD *)(v19 + 8) == -1 )
        goto LABEL_18;
LABEL_13:
      v20 = v17 + i;
      ++v17;
    }
    if ( *(_DWORD *)(v19 + 8) != v6 )
      goto LABEL_13;
    *(_QWORD *)v19 = 0;
    *(_DWORD *)(v19 + 8) = -2;
    v22 = *(_DWORD *)(a1 + 144);
    ++*(_DWORD *)(a1 + 148);
    *(_DWORD *)(a1 + 144) = (2 * (v22 >> 1) - 2) | v22 & 1;
LABEL_18:
    if ( (*(_BYTE *)(a1 + 352) & 1) != 0 )
    {
      v23 = a1 + 360;
      v24 = 7;
LABEL_20:
      v25 = v24 & (37 * v8);
      v26 = (int *)(v23 + 24LL * v25);
      v27 = *v26;
      if ( v8 == *v26 )
      {
LABEL_21:
        *v26 = -2;
        v28 = *(_DWORD *)(a1 + 352);
        ++*(_DWORD *)(a1 + 356);
        *(_DWORD *)(a1 + 352) = (2 * (v28 >> 1) - 2) | v28 & 1;
      }
      else
      {
        v88 = 1;
        while ( v27 != -1 )
        {
          v89 = v88 + 1;
          v25 = v24 & (v88 + v25);
          v26 = (int *)(v23 + 24LL * v25);
          v27 = *v26;
          if ( v8 == *v26 )
            goto LABEL_21;
          v88 = v89;
        }
      }
    }
    else
    {
      v83 = *(_DWORD *)(a1 + 368);
      v23 = *(_QWORD *)(a1 + 360);
      if ( v83 )
      {
        v24 = v83 - 1;
        goto LABEL_20;
      }
    }
    if ( (*(_BYTE *)(a1 + 560) & 1) != 0 )
    {
      v29 = a1 + 568;
      v30 = 7;
LABEL_24:
      v31 = v30 & (37 * v8);
      v32 = (int *)(v29 + 8LL * v31);
      v33 = *v32;
      if ( v8 == *v32 )
      {
LABEL_25:
        *v32 = -2;
        v34 = *(_DWORD *)(a1 + 560);
        ++*(_DWORD *)(a1 + 564);
        *(_DWORD *)(a1 + 560) = (2 * (v34 >> 1) - 2) | v34 & 1;
      }
      else
      {
        v100 = 1;
        while ( v33 != -1 )
        {
          v101 = v100 + 1;
          v31 = v30 & (v100 + v31);
          v32 = (int *)(v29 + 8LL * v31);
          v33 = *v32;
          if ( v8 == *v32 )
            goto LABEL_25;
          v100 = v101;
        }
      }
    }
    else
    {
      v82 = *(_DWORD *)(a1 + 576);
      v29 = *(_QWORD *)(a1 + 568);
      if ( v82 )
      {
        v30 = v82 - 1;
        goto LABEL_24;
      }
    }
    if ( (*(_BYTE *)(a1 + 640) & 1) != 0 )
    {
      v35 = a1 + 648;
      v36 = 7;
LABEL_28:
      v37 = v36 & (37 * v8);
      v38 = (int *)(v35 + 12LL * v37);
      v39 = *v38;
      if ( v8 == *v38 )
      {
LABEL_29:
        *v38 = -2;
        v40 = *(_DWORD *)(a1 + 640);
        ++*(_DWORD *)(a1 + 644);
        *(_DWORD *)(a1 + 640) = (2 * (v40 >> 1) - 2) | v40 & 1;
      }
      else
      {
        v98 = 1;
        while ( v39 != -1 )
        {
          v99 = v98 + 1;
          v37 = v36 & (v98 + v37);
          v38 = (int *)(v35 + 12LL * v37);
          v39 = *v38;
          if ( v8 == *v38 )
            goto LABEL_29;
          v98 = v99;
        }
      }
    }
    else
    {
      v81 = *(_DWORD *)(a1 + 656);
      v35 = *(_QWORD *)(a1 + 648);
      if ( v81 )
      {
        v36 = v81 - 1;
        goto LABEL_28;
      }
    }
    if ( (*(_BYTE *)(a1 + 752) & 1) != 0 )
    {
      v41 = a1 + 760;
      v42 = 7;
LABEL_32:
      v43 = v42 & (37 * v8);
      v44 = (int *)(v41 + 8LL * v43);
      v45 = *v44;
      if ( v8 == *v44 )
      {
LABEL_33:
        *v44 = -2;
        v46 = *(_DWORD *)(a1 + 752);
        ++*(_DWORD *)(a1 + 756);
        *(_DWORD *)(a1 + 752) = (2 * (v46 >> 1) - 2) | v46 & 1;
      }
      else
      {
        v104 = 1;
        while ( v45 != -1 )
        {
          v105 = v104 + 1;
          v43 = v42 & (v104 + v43);
          v44 = (int *)(v41 + 8LL * v43);
          v45 = *v44;
          if ( v8 == *v44 )
            goto LABEL_33;
          v104 = v105;
        }
      }
    }
    else
    {
      v80 = *(_DWORD *)(a1 + 768);
      v41 = *(_QWORD *)(a1 + 760);
      if ( v80 )
      {
        v42 = v80 - 1;
        goto LABEL_32;
      }
    }
    if ( (*(_BYTE *)(a1 + 832) & 1) != 0 )
    {
      v47 = a1 + 840;
      v48 = 7;
LABEL_36:
      v49 = v48 & (37 * v8);
      v50 = (int *)(v47 + 8LL * v49);
      v51 = *v50;
      if ( v8 == *v50 )
      {
LABEL_37:
        *v50 = -2;
        v52 = *(_DWORD *)(a1 + 832);
        ++*(_DWORD *)(a1 + 836);
        *(_DWORD *)(a1 + 832) = (2 * (v52 >> 1) - 2) | v52 & 1;
      }
      else
      {
        v102 = 1;
        while ( v51 != -1 )
        {
          v103 = v102 + 1;
          v49 = v48 & (v102 + v49);
          v50 = (int *)(v47 + 8LL * v49);
          v51 = *v50;
          if ( v8 == *v50 )
            goto LABEL_37;
          v102 = v103;
        }
      }
    }
    else
    {
      v79 = *(_DWORD *)(a1 + 848);
      v47 = *(_QWORD *)(a1 + 840);
      if ( v79 )
      {
        v48 = v79 - 1;
        goto LABEL_36;
      }
    }
    if ( (*(_BYTE *)(a1 + 912) & 1) != 0 )
    {
      v53 = a1 + 920;
      v54 = 7;
LABEL_40:
      v55 = v54 & (37 * v8);
      v56 = (int *)(v53 + 12LL * v55);
      v57 = *v56;
      if ( v8 == *v56 )
      {
LABEL_41:
        *v56 = -2;
        v58 = *(_DWORD *)(a1 + 912);
        ++*(_DWORD *)(a1 + 916);
        *(_DWORD *)(a1 + 912) = (2 * (v58 >> 1) - 2) | v58 & 1;
      }
      else
      {
        v92 = 1;
        while ( v57 != -1 )
        {
          v93 = v92 + 1;
          v55 = v54 & (v92 + v55);
          v56 = (int *)(v53 + 12LL * v55);
          v57 = *v56;
          if ( v8 == *v56 )
            goto LABEL_41;
          v92 = v93;
        }
      }
    }
    else
    {
      v78 = *(_DWORD *)(a1 + 928);
      v53 = *(_QWORD *)(a1 + 920);
      if ( v78 )
      {
        v54 = v78 - 1;
        goto LABEL_40;
      }
    }
    if ( (*(_BYTE *)(a1 + 1024) & 1) != 0 )
    {
      v59 = a1 + 1032;
      v60 = 7;
LABEL_44:
      v61 = v60 & (37 * v8);
      v62 = (int *)(v59 + 8LL * v61);
      v63 = *v62;
      if ( v8 == *v62 )
      {
LABEL_45:
        *v62 = -2;
        v64 = *(_DWORD *)(a1 + 1024);
        ++*(_DWORD *)(a1 + 1028);
        *(_DWORD *)(a1 + 1024) = (2 * (v64 >> 1) - 2) | v64 & 1;
      }
      else
      {
        v90 = 1;
        while ( v63 != -1 )
        {
          v91 = v90 + 1;
          v61 = v60 & (v90 + v61);
          v62 = (int *)(v59 + 8LL * v61);
          v63 = *v62;
          if ( v8 == *v62 )
            goto LABEL_45;
          v90 = v91;
        }
      }
    }
    else
    {
      v77 = *(_DWORD *)(a1 + 1040);
      v59 = *(_QWORD *)(a1 + 1032);
      if ( v77 )
      {
        v60 = v77 - 1;
        goto LABEL_44;
      }
    }
    if ( (*(_BYTE *)(a1 + 1104) & 1) != 0 )
    {
      v65 = a1 + 1112;
      v66 = 7;
LABEL_48:
      v67 = v66 & (37 * v8);
      v68 = (int *)(v65 + 12LL * v67);
      v69 = *v68;
      if ( v8 == *v68 )
      {
LABEL_49:
        *v68 = -2;
        v70 = *(_DWORD *)(a1 + 1104);
        ++*(_DWORD *)(a1 + 1108);
        *(_DWORD *)(a1 + 1104) = (2 * (v70 >> 1) - 2) | v70 & 1;
      }
      else
      {
        v96 = 1;
        while ( v69 != -1 )
        {
          v97 = v96 + 1;
          v67 = v66 & (v96 + v67);
          v68 = (int *)(v65 + 12LL * v67);
          v69 = *v68;
          if ( v8 == *v68 )
            goto LABEL_49;
          v96 = v97;
        }
      }
    }
    else
    {
      v76 = *(_DWORD *)(a1 + 1120);
      v65 = *(_QWORD *)(a1 + 1112);
      if ( v76 )
      {
        v66 = v76 - 1;
        goto LABEL_48;
      }
    }
    if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    {
      v71 = a1 + 1224;
      v72 = 7;
LABEL_52:
      result = v72 & (unsigned int)(37 * v8);
      v73 = (int *)(v71 + 8 * result);
      v74 = *v73;
      if ( v8 == *v73 )
      {
LABEL_53:
        *v73 = -2;
        v75 = *(_DWORD *)(a1 + 1216);
        ++*(_DWORD *)(a1 + 1220);
        result = (2 * (v75 >> 1) - 2) | v75 & 1;
        *(_DWORD *)(a1 + 1216) = result;
      }
      else
      {
        v94 = 1;
        while ( v74 != -1 )
        {
          v95 = v94 + 1;
          result = v72 & (unsigned int)(v94 + result);
          v73 = (int *)(v71 + 8LL * (unsigned int)result);
          v74 = *v73;
          if ( v8 == *v73 )
            goto LABEL_53;
          v94 = v95;
        }
      }
    }
    else
    {
      result = *(unsigned int *)(a1 + 1232);
      v71 = *(_QWORD *)(a1 + 1224);
      if ( (_DWORD)result )
      {
        v72 = result - 1;
        goto LABEL_52;
      }
    }
    ++v6;
  }
  while ( v123 != v6 );
  return result;
}
