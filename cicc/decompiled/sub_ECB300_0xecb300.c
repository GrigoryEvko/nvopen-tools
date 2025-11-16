// Function: sub_ECB300
// Address: 0xecb300
//
__int64 __fastcall sub_ECB300(__int64 a1, char a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  _DWORD *v6; // rdi
  unsigned int v7; // r13d
  size_t v8; // r14
  _BYTE *v9; // r15
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // r14
  __int64 v25; // rdx
  int v26; // r14d
  __int64 v27; // r12
  int v28; // r13d
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  _QWORD *v32; // rdx
  size_t v33; // rcx
  const char *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // r12
  _QWORD *v37; // rax
  int v38; // eax
  unsigned __int64 v39; // rax
  _BYTE *v40; // r13
  size_t v41; // r8
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  const char *v46; // rax
  __int64 result; // rax
  const char *v48; // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rax
  void *v55; // r13
  const void *v56; // rdi
  __int64 v57; // rdi
  _BYTE *v58; // r13
  size_t v59; // r8
  _QWORD *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rdx
  int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rdi
  int v66; // r13d
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // r13
  const char *v73; // rax
  __int64 v74; // rdi
  __int64 v75; // r13
  __int64 v76; // rdi
  __int64 v77; // rdi
  const char *v78; // rax
  __int64 v79; // rax
  __int64 v80; // r9
  _BYTE *v81; // r12
  __int64 v82; // rsi
  __int64 v83; // r13
  __int64 v84; // rdi
  __int64 v85; // r13
  int v86; // eax
  __int64 v87; // rdi
  __int64 v88; // rax
  int v89; // esi
  int v90; // eax
  __int64 v91; // rax
  _QWORD *v92; // rdi
  __int64 v93; // rax
  __int64 v94; // rdi
  size_t v95; // rdx
  void *v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rax
  _QWORD *v99; // rdi
  __int64 v100; // rdi
  __int64 v101; // r13
  const char *v102; // rax
  __int64 v103; // rdi
  char v104; // al
  __int64 v105; // rdi
  __int64 v106; // r13
  char v107; // al
  __int64 v108; // rdi
  __int64 v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rdi
  __int64 v112; // rax
  __int64 v113; // rdi
  size_t v114; // rdx
  const char *v115; // rax
  void *v116; // rax
  __int64 v117; // rax
  size_t v118; // [rsp+8h] [rbp-158h]
  size_t n; // [rsp+10h] [rbp-150h]
  size_t na; // [rsp+10h] [rbp-150h]
  char v121; // [rsp+18h] [rbp-148h]
  int v123; // [rsp+28h] [rbp-138h]
  __int64 v124; // [rsp+28h] [rbp-138h]
  __int64 v125; // [rsp+30h] [rbp-130h]
  __int64 v126; // [rsp+30h] [rbp-130h]
  unsigned int v127; // [rsp+30h] [rbp-130h]
  __int64 v128; // [rsp+38h] [rbp-128h]
  int v129; // [rsp+38h] [rbp-128h]
  __int64 v130; // [rsp+48h] [rbp-118h] BYREF
  __int64 v131; // [rsp+50h] [rbp-110h] BYREF
  __int64 v132; // [rsp+58h] [rbp-108h] BYREF
  void *s1; // [rsp+60h] [rbp-100h] BYREF
  size_t v134; // [rsp+68h] [rbp-F8h]
  const char *v135; // [rsp+70h] [rbp-F0h] BYREF
  size_t v136; // [rsp+78h] [rbp-E8h]
  _QWORD *v137; // [rsp+80h] [rbp-E0h] BYREF
  size_t v138; // [rsp+88h] [rbp-D8h]
  _QWORD v139[2]; // [rsp+90h] [rbp-D0h] BYREF
  _QWORD v140[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE *v141; // [rsp+B0h] [rbp-B0h]
  size_t v142; // [rsp+B8h] [rbp-A8h]
  __int16 v143; // [rsp+C0h] [rbp-A0h]
  void *v144; // [rsp+D0h] [rbp-90h] BYREF
  size_t v145; // [rsp+D8h] [rbp-88h]
  const char *v146; // [rsp+E0h] [rbp-80h]
  __int16 v147; // [rsp+F0h] [rbp-70h]
  const char *v148; // [rsp+100h] [rbp-60h] BYREF
  size_t v149; // [rsp+108h] [rbp-58h]
  _QWORD v150[2]; // [rsp+110h] [rbp-50h] BYREF
  __int16 v151; // [rsp+120h] [rbp-40h]

  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v128 = sub_ECD690(v4);
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v6 = *(_DWORD **)(a1 + 8);
  if ( **(_DWORD **)(v5 + 8) == 3 )
  {
    v53 = sub_ECD7B0(v6);
    if ( *(_DWORD *)v53 == 2 )
    {
      v9 = *(_BYTE **)(v53 + 8);
      v8 = *(_QWORD *)(v53 + 16);
    }
    else
    {
      v8 = *(_QWORD *)(v53 + 16);
      v9 = *(_BYTE **)(v53 + 8);
      if ( v8 )
      {
        v54 = v8 - 1;
        if ( v8 == 1 )
          v54 = 1;
        ++v9;
        v8 = v54 - 1;
      }
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
LABEL_25:
    s1 = 0;
    v134 = 0;
    v130 = 0;
    v135 = 0;
    v136 = 0;
    v131 = 0;
    v132 = -1;
    if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".rodata", 7u) )
      goto LABEL_26;
    if ( v8 == 8 )
    {
      if ( *(_QWORD *)v9 == 0x31617461646F722ELL )
      {
LABEL_26:
        v127 = 2;
        goto LABEL_27;
      }
    }
    else if ( v8 == 5 && (*(_DWORD *)v9 == 1852401198 && v9[4] == 105 || *(_DWORD *)v9 == 1768843566 && v9[4] == 116) )
    {
      goto LABEL_66;
    }
    if ( !(unsigned __int8)sub_EC9EC0(v9, v8, ".text", 5u) )
    {
      if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".data", 5u)
        || v8 == 6 && *(_DWORD *)v9 == 1952539694 && *((_WORD *)v9 + 2) == 12641
        || (unsigned __int8)sub_EC9EC0(v9, v8, ".bss", 4u)
        || (unsigned __int8)sub_EC9EC0(v9, v8, ".init_array", 0xBu)
        || (unsigned __int8)sub_EC9EC0(v9, v8, ".fini_array", 0xBu)
        || (unsigned __int8)sub_EC9EC0(v9, v8, ".preinit_array", 0xEu) )
      {
        v127 = 3;
      }
      else if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".tdata", 6u) || (unsigned __int8)sub_EC9EC0(v9, v8, ".tbss", 5u) )
      {
        v127 = 1027;
      }
      else
      {
        v127 = 0;
      }
LABEL_27:
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
      {
        LOBYTE(v27) = 0;
        n = 0;
        v121 = 0;
        v123 = 0;
        goto LABEL_29;
      }
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      if ( a2
        && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
      {
        LOBYTE(v27) = sub_ECD870(*(_QWORD *)(a1 + 8), &v131);
        if ( (_BYTE)v27 )
          return 1;
        if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
        {
          v121 = 0;
          n = 0;
          v123 = 0;
          goto LABEL_29;
        }
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      }
      v50 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
      v51 = *(_QWORD *)(a1 + 8);
      if ( **(_DWORD **)(v50 + 8) != 3 )
      {
        if ( **(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v51 + 40LL))(v51) + 8) != 38 )
        {
          v52 = *(_QWORD *)(a1 + 8);
          v148 = "expected string";
          v151 = 259;
          return sub_ECE0E0(v52, &v148, 0, 0);
        }
        v66 = 0;
        while ( 1 )
        {
          if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 38 )
          {
LABEL_198:
            v123 = v66;
            goto LABEL_199;
          }
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
          if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2 )
            goto LABEL_160;
          v67 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
          if ( *(_DWORD *)v67 == 2 )
          {
            v71 = *(_QWORD *)(v67 + 16);
            v70 = *(_QWORD *)(v67 + 8);
            if ( v71 != 5 )
              goto LABEL_158;
          }
          else
          {
            v68 = *(_QWORD *)(v67 + 16);
            if ( !v68 )
              goto LABEL_160;
            v69 = v68 - 1;
            if ( !v69 )
              v69 = 1;
            v70 = *(_QWORD *)(v67 + 8) + 1LL;
            v71 = v69 - 1;
            if ( v71 != 5 )
            {
LABEL_158:
              if ( v71 == 9 )
              {
                if ( *(_QWORD *)v70 != 0x74736E6963657865LL || *(_BYTE *)(v70 + 8) != 114 )
                  goto LABEL_160;
                v66 |= 4u;
              }
              else
              {
                if ( v71 != 3 || *(_WORD *)v70 != 27764 || *(_BYTE *)(v70 + 2) != 115 )
                  goto LABEL_160;
                v66 |= 0x400u;
              }
              goto LABEL_166;
            }
          }
          if ( *(_DWORD *)v70 == 1869376609 && *(_BYTE *)(v70 + 4) == 99 )
          {
            v66 |= 2u;
          }
          else
          {
            if ( *(_DWORD *)v70 != 1953067639 || *(_BYTE *)(v70 + 4) != 101 )
              goto LABEL_160;
            v66 |= 1u;
          }
LABEL_166:
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
          if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
            goto LABEL_198;
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        }
      }
      v79 = sub_ECD7B0(v51);
      v80 = *(_QWORD *)(v79 + 16);
      if ( v80 )
      {
        --v80;
        v81 = (_BYTE *)(*(_QWORD *)(v79 + 8) + 1LL);
        if ( !v80 )
          v80 = 1;
        v82 = v80 - 1;
      }
      else
      {
        v81 = *(_BYTE **)(v79 + 8);
        v82 = 0;
      }
      na = *(_QWORD *)(v79 + 8);
      v124 = v80;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v83 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v121 = sub_C93C90((__int64)v81, v82, 0, (unsigned __int64 *)&v148);
      if ( v121 || v148 != (const char *)(unsigned int)v148 )
      {
        if ( v81 == (_BYTE *)(na + v124) )
        {
          v123 = 0;
LABEL_199:
          v127 |= v123;
          LOBYTE(v27) = 0;
          v121 = 0;
          v129 = 0;
          goto LABEL_200;
        }
        v121 = 0;
        v89 = 0;
        do
        {
          switch ( *v81 )
          {
            case '?':
              v121 = 1;
              break;
            case 'G':
              v89 |= 0x200u;
              break;
            case 'M':
              v89 |= 0x10u;
              break;
            case 'R':
              if ( *(_DWORD *)(v83 + 68) == 12 )
                v89 |= 0x100000u;
              else
                v89 |= 0x200000u;
              break;
            case 'S':
              v89 |= 0x20u;
              break;
            case 'T':
              v89 |= 0x400u;
              break;
            case 'a':
              v89 |= 2u;
              break;
            case 'c':
              if ( *(_DWORD *)(v83 + 56) != 40 )
                goto LABEL_160;
              goto LABEL_219;
            case 'd':
              if ( *(_DWORD *)(v83 + 56) != 40 )
                goto LABEL_160;
              goto LABEL_225;
            case 'e':
              v89 |= 0x80000000;
              break;
            case 'l':
              if ( *(_DWORD *)(v83 + 56) != 39 )
                goto LABEL_160;
              goto LABEL_225;
            case 'o':
              LOBYTE(v89) = v89 | 0x80;
              break;
            case 's':
              if ( *(_DWORD *)(v83 + 56) != 12 )
                goto LABEL_160;
LABEL_225:
              v89 |= 0x10000000u;
              break;
            case 'w':
              v89 |= 1u;
              break;
            case 'x':
              v89 |= 4u;
              break;
            case 'y':
              v90 = *(_DWORD *)(v83 + 56);
              if ( (unsigned int)(v90 - 36) > 1 && (unsigned int)(v90 - 1) > 1 && (unsigned int)(v90 - 3) > 2 )
                goto LABEL_160;
LABEL_219:
              v89 |= 0x20000000u;
              break;
            default:
              goto LABEL_160;
          }
          ++v81;
        }
        while ( (_BYTE *)(na + v124) != v81 );
        v123 = v89;
      }
      else
      {
        v123 = (int)v148;
        if ( (_DWORD)v148 == -1 )
        {
LABEL_160:
          HIBYTE(v151) = 1;
          v48 = "unknown flag";
          goto LABEL_69;
        }
      }
      v127 |= v123;
      LOBYTE(v27) = (v127 & 0x200) != 0;
      if ( (v127 & 0x200) != 0 && v121 )
      {
        v84 = *(_QWORD *)(a1 + 8);
        v148 = "Section cannot specifiy a group name while also acting as a member of the last group";
        v151 = 259;
        return sub_ECE0E0(v84, &v148, 0, 0);
      }
      v129 = v127 & 0x10;
LABEL_200:
      v85 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
      if ( **(_DWORD **)(v85 + 8) != 26 )
        goto LABEL_208;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v86 = **(_DWORD **)(v85 + 8);
      if ( v86 == 46 || v86 == 37 )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        if ( **(_DWORD **)(v85 + 8) == 4 )
        {
          v93 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
          v94 = *(_QWORD *)(a1 + 8);
          v95 = *(_QWORD *)(v93 + 16);
          v96 = *(void **)(v93 + 8);
          v134 = v95;
          s1 = v96;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v94 + 184LL))(v94);
          goto LABEL_208;
        }
      }
      else if ( v86 != 3 )
      {
        v87 = *(_QWORD *)(a1 + 8);
        if ( *(_BYTE *)(v85 + 113) )
          v148 = "expected '@<type>', '%<type>' or \"<type>\"";
        else
          v148 = "expected '%<type>' or \"<type>\"";
        v151 = 259;
LABEL_207:
        if ( (unsigned __int8)sub_ECE0E0(v87, &v148, 0, 0) )
          return 1;
LABEL_208:
        v88 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
        if ( !v134 )
        {
          v6 = *(_DWORD **)(a1 + 8);
          if ( v129 )
          {
            HIBYTE(v151) = 1;
            v46 = "Mergeable section must specify the type";
            goto LABEL_61;
          }
          if ( (_BYTE)v27 )
          {
            HIBYTE(v151) = 1;
            v46 = "Group section must specify the type";
            goto LABEL_61;
          }
          if ( **(_DWORD **)(v88 + 8) != 9 )
          {
            HIBYTE(v151) = 1;
            v46 = "expected end of directive";
            goto LABEL_61;
          }
          if ( (v127 & 0x80u) == 0 )
          {
            n = 0;
LABEL_178:
            v75 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v6 + 40LL))(v6);
            if ( **(_DWORD **)(v75 + 8) != 26 )
              goto LABEL_29;
            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
            v76 = *(_QWORD *)(a1 + 8);
            v144 = 0;
            v145 = 0;
            if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v76 + 192LL))(v76, &v144) )
            {
              v77 = *(_QWORD *)(a1 + 8);
              v148 = "expected identifier";
              v151 = 259;
            }
            else
            {
              v77 = *(_QWORD *)(a1 + 8);
              if ( v145 == 6 && *(_DWORD *)v144 == 1902734965 && *((_WORD *)v144 + 2) == 25973 )
              {
                if ( **(_DWORD **)(v75 + 8) == 26 )
                {
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v77 + 184LL))(v77);
                  result = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(
                             *(_QWORD *)(a1 + 8),
                             &v132);
                  if ( (_BYTE)result )
                    return result;
                  v77 = *(_QWORD *)(a1 + 8);
                  if ( v132 < 0 )
                  {
                    HIBYTE(v151) = 1;
                    v78 = "unique id must be positive";
                  }
                  else
                  {
                    if ( (unsigned int)v132 == v132 && v132 != 0xFFFFFFFFLL )
                      goto LABEL_29;
                    HIBYTE(v151) = 1;
                    v78 = "unique id is too large";
                  }
                }
                else
                {
                  HIBYTE(v151) = 1;
                  v78 = "expected commma";
                }
              }
              else
              {
                HIBYTE(v151) = 1;
                v78 = "expected 'unique'";
              }
              v148 = v78;
              LOBYTE(v151) = 3;
            }
            if ( (unsigned __int8)sub_ECE0E0(v77, &v148, 0, 0) )
              return 1;
LABEL_29:
            if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8))
                             + 8) == 9 )
            {
              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
              switch ( v134 )
              {
                case 0uLL:
                  if ( v8 <= 4 || *(_DWORD *)v9 != 1953459758 || (v28 = 7, v9[4] != 101) )
                  {
                    if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".init_array", 0xBu) )
                    {
LABEL_35:
                      v28 = 14;
                      goto LABEL_36;
                    }
                    if ( !(unsigned __int8)sub_EC9EC0(v9, v8, ".bss", 4u)
                      && !(unsigned __int8)sub_EC9EC0(v9, v8, ".tbss", 5u) )
                    {
                      v28 = 15;
                      if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".fini_array", 0xBu) )
                        goto LABEL_36;
                      if ( (unsigned __int8)sub_EC9EC0(v9, v8, ".preinit_array", 0xEu) )
                        goto LABEL_119;
LABEL_285:
                      v28 = 1;
                      goto LABEL_36;
                    }
LABEL_139:
                    v28 = 8;
                    goto LABEL_36;
                  }
                  goto LABEL_36;
                case 0xAuLL:
                  if ( *(_QWORD *)s1 == 0x7272615F74696E69LL && *((_WORD *)s1 + 4) == 31073 )
                    goto LABEL_35;
                  if ( *(_QWORD *)s1 == 0x7272615F696E6966LL && *((_WORD *)s1 + 4) == 31073 )
                  {
                    v28 = 15;
                    goto LABEL_36;
                  }
                  break;
                case 0xDuLL:
                  if ( *(_QWORD *)s1 == 0x5F74696E69657270LL
                    && *((_DWORD *)s1 + 2) == 1634890337
                    && *((_BYTE *)s1 + 12) == 121 )
                  {
LABEL_119:
                    v28 = 16;
                    goto LABEL_36;
                  }
                  break;
                case 6uLL:
                  if ( *(_DWORD *)s1 == 1768058734 && *((_WORD *)s1 + 2) == 29556 )
                    goto LABEL_139;
                  v28 = 1879048193;
                  if ( !memcmp(s1, "unwind", 6u) )
                    goto LABEL_36;
                  break;
                case 8uLL:
                  if ( *(_QWORD *)s1 == 0x73746962676F7270LL )
                    goto LABEL_285;
                  break;
                case 4uLL:
                  v28 = 7;
                  if ( *(_DWORD *)s1 == 1702129518 )
                    goto LABEL_36;
                  break;
              }
              v55 = s1;
              v118 = v134;
              if ( sub_9691B0(s1, v134, "llvm_odrtab", 11) )
              {
                v28 = 1879002112;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_linker_options", 19) )
              {
                v28 = 1879002113;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_call_graph_profile", 23) )
              {
                v28 = 1879002121;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_dependent_libraries", 24) )
              {
                v28 = 1879002116;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_sympart", 12) )
              {
                v28 = 1879002117;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_bb_addr_map", 16) )
              {
                v28 = 1879002122;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_offloading", 15) )
              {
                v28 = 1879002123;
                goto LABEL_36;
              }
              if ( sub_9691B0(v55, v118, "llvm_lto", 8) )
              {
                v28 = 1879002124;
                goto LABEL_36;
              }
              v56 = v55;
              v28 = 1879002125;
              if ( sub_9691B0(v56, v118, "llvm_jt_sizes", 13)
                || !sub_C93C90((__int64)s1, v134, 0, (unsigned __int64 *)&v148)
                && (v28 = (int)v148, v148 == (const char *)(unsigned int)v148) )
              {
LABEL_36:
                if ( v121 )
                {
                  v29 = *(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8))
                                              + 288)
                                  + 8LL);
                  if ( v29 )
                  {
                    v30 = *(_QWORD *)(v29 + 168);
                    v31 = v30 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (v30 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                    {
                      if ( (*(_BYTE *)(v31 + 8) & 1) != 0 )
                      {
                        v32 = *(_QWORD **)(v31 - 8);
                        v33 = *v32;
                        v34 = (const char *)(v32 + 3);
                      }
                      else
                      {
                        v33 = 0;
                        v34 = 0;
                      }
                      v127 |= 0x200u;
                      v135 = v34;
                      v136 = v33;
                      v27 = (v30 >> 2) & 1;
                    }
                  }
                }
                v35 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
                v151 = 261;
                v148 = v135;
                v144 = v9;
                v149 = v136;
                v147 = 261;
                v145 = v8;
                v36 = sub_E71CB0(v35, (size_t *)&v144, v28, v127, v130, (__int64)&v148, v27, v132, n);
                v37 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
                sub_E9A6D0(v37, v36, v131);
                if ( !v134 || *(_DWORD *)(v36 + 148) == v28 )
                {
LABEL_53:
                  if ( v123 )
                  {
                    v45 = *(unsigned int *)(v36 + 152);
                    if ( v127 == (_DWORD)v45 )
                    {
                      v44 = v130;
                      goto LABEL_100;
                    }
                  }
                  else
                  {
                    v44 = v130;
                    if ( v130 )
                    {
                      v45 = *(unsigned int *)(v36 + 152);
                      if ( v127 == (_DWORD)v45 )
                        goto LABEL_100;
                    }
                    else
                    {
                      if ( !v134 )
                        goto LABEL_102;
                      v45 = *(unsigned int *)(v36 + 152);
                      if ( v127 == (_DWORD)v45 )
                      {
                        v44 = 0;
                        goto LABEL_100;
                      }
                    }
                  }
                  if ( !v45 )
                  {
                    LOBYTE(v150[0]) = 48;
                    v58 = v150;
                    v137 = v139;
LABEL_108:
                    v59 = 1;
                    LOBYTE(v139[0]) = *v58;
                    v60 = v139;
                    goto LABEL_109;
                  }
                  v58 = (char *)v150 + 1;
                  do
                  {
                    --v58;
                    v62 = v45 & 0xF;
                    v45 >>= 4;
                    *v58 = a0123456789abcd_10[v62];
                  }
                  while ( v45 );
                  v59 = (char *)v150 + 1 - v58;
                  v137 = v139;
                  v144 = (void *)((char *)v150 + 1 - v58);
                  if ( (unsigned __int64)((char *)v150 + 1 - v58) > 0xF )
                  {
                    v91 = sub_22409D0(&v137, &v144, 0);
                    v59 = (char *)v150 + 1 - v58;
                    v137 = (_QWORD *)v91;
                    v92 = (_QWORD *)v91;
                    v139[0] = v144;
                  }
                  else
                  {
                    if ( v59 == 1 )
                      goto LABEL_108;
                    if ( !v59 )
                    {
                      v60 = v139;
                      goto LABEL_109;
                    }
                    v92 = v139;
                  }
                  memcpy(v92, v58, v59);
                  v59 = (size_t)v144;
                  v60 = v137;
LABEL_109:
                  v138 = v59;
                  *((_BYTE *)v60 + v59) = 0;
                  v61 = *(_QWORD *)(a1 + 8);
                  v140[0] = "changed section flags for ";
                  v144 = v140;
                  v146 = ", expected: 0x";
                  v143 = 1283;
                  v150[0] = &v137;
                  v147 = 770;
                  v141 = v9;
                  v142 = v8;
                  v148 = (const char *)&v144;
                  v151 = 1026;
                  sub_ECDA70(v61, a3, &v148, 0, 0);
                  if ( v137 != v139 )
                    j_j___libc_free_0(v137, v139[0] + 1LL);
                  v44 = v130;
                  if ( v123 || v130 )
                    goto LABEL_100;
                  if ( v134 )
                  {
                    v44 = 0;
LABEL_100:
                    if ( *(_DWORD *)(v36 + 160) != v44 )
                    {
                      LODWORD(v150[0]) = *(_DWORD *)(v36 + 160);
                      v143 = 1283;
                      v57 = *(_QWORD *)(a1 + 8);
                      v140[0] = "changed section entsize for ";
                      v146 = ", expected: ";
                      v147 = 770;
                      v144 = v140;
                      v151 = 2306;
                      v148 = (const char *)&v144;
                      v141 = v9;
                      v142 = v8;
                      sub_ECDA70(v57, a3, &v148, 0, 0);
                    }
                  }
LABEL_102:
                  if ( !*(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                                 + 1793) )
                    return 0;
                  if ( (*(_DWORD *)(v36 + 152) & 6) != 6 )
                    return 0;
                  v64 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
                  v148 = (const char *)v36;
                  if ( !(unsigned __int8)sub_EAB3D0(v64 + 1800, (__int64 *)&v148)
                    || *(_WORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                                + 1904) > 2u )
                  {
                    return 0;
                  }
                  v65 = *(_QWORD *)(a1 + 8);
                  v148 = "DWARF2 only supports one section per compilation unit";
                  v151 = 259;
                  (*(void (__fastcall **)(__int64, __int64, const char **, _QWORD, _QWORD))(*(_QWORD *)v65 + 168LL))(
                    v65,
                    a3,
                    &v148,
                    0,
                    0);
                  return 0;
                }
                v38 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                                + 56);
                if ( v38 == 39 )
                {
                  if ( v8 != 9 )
                    goto LABEL_48;
                  if ( *(_QWORD *)v9 != 0x6D6172665F68652ELL || (v63 = 0, v9[8] != 101) )
                    v63 = 1;
                  if ( v63 )
                    goto LABEL_48;
                }
                else if ( (unsigned int)(v38 - 16) > 3
                       || v8 <= 6
                       || *(_DWORD *)v9 != 1650811950
                       || *((_WORD *)v9 + 2) != 26485
                       || v9[6] != 95 )
                {
                  goto LABEL_48;
                }
                if ( v28 == 1 )
                  goto LABEL_53;
LABEL_48:
                v39 = *(unsigned int *)(v36 + 148);
                if ( !*(_DWORD *)(v36 + 148) )
                {
                  LOBYTE(v150[0]) = 48;
                  v40 = v150;
                  v137 = v139;
LABEL_50:
                  v41 = 1;
                  LOBYTE(v139[0]) = *v40;
                  v42 = v139;
                  goto LABEL_51;
                }
                v40 = (char *)v150 + 1;
                do
                {
                  --v40;
                  v97 = v39 & 0xF;
                  v39 >>= 4;
                  *v40 = a0123456789abcd_10[v97];
                }
                while ( v39 );
                v41 = (char *)v150 + 1 - v40;
                v137 = v139;
                v144 = (void *)((char *)v150 + 1 - v40);
                if ( (unsigned __int64)((char *)v150 + 1 - v40) <= 0xF )
                {
                  if ( v41 == 1 )
                    goto LABEL_50;
                  if ( !v41 )
                  {
                    v42 = v139;
LABEL_51:
                    v138 = v41;
                    *((_BYTE *)v42 + v41) = 0;
                    v43 = *(_QWORD *)(a1 + 8);
                    v140[0] = "changed section type for ";
                    v144 = v140;
                    v146 = ", expected: 0x";
                    v147 = 770;
                    v150[0] = &v137;
                    v148 = (const char *)&v144;
                    v143 = 1283;
                    v141 = v9;
                    v142 = v8;
                    v151 = 1026;
                    sub_ECDA70(v43, a3, &v148, 0, 0);
                    if ( v137 != v139 )
                      j_j___libc_free_0(v137, v139[0] + 1LL);
                    goto LABEL_53;
                  }
                  v99 = v139;
                }
                else
                {
                  v98 = sub_22409D0(&v137, &v144, 0);
                  v41 = (char *)v150 + 1 - v40;
                  v137 = (_QWORD *)v98;
                  v99 = (_QWORD *)v98;
                  v139[0] = v144;
                }
                memcpy(v99, v40, v41);
                v41 = (size_t)v144;
                v42 = v137;
                goto LABEL_51;
              }
              HIBYTE(v151) = 1;
              v48 = "unknown section type";
            }
            else
            {
              HIBYTE(v151) = 1;
              v48 = "expected end of directive";
            }
LABEL_69:
            v49 = *(_QWORD *)(a1 + 8);
            v148 = v48;
            LOBYTE(v151) = 3;
            return sub_ECE0E0(v49, &v148, 0, 0);
          }
          goto LABEL_273;
        }
        if ( v129 )
        {
          if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
          {
            (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
            if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 256LL))(
                   *(_QWORD *)(a1 + 8),
                   &v130) )
            {
              return 1;
            }
            if ( v130 > 0 )
              goto LABEL_171;
            v100 = *(_QWORD *)(a1 + 8);
            v148 = "entry size must be positive";
            v151 = 259;
          }
          else
          {
            v100 = *(_QWORD *)(a1 + 8);
            v148 = "expected the entry size";
            v151 = 259;
          }
          if ( (unsigned __int8)sub_ECE0E0(v100, &v148, 0, 0) )
            return 1;
        }
LABEL_171:
        if ( (v127 & 0x80u) == 0 )
          goto LABEL_172;
        v6 = *(_DWORD **)(a1 + 8);
LABEL_273:
        v101 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v6 + 40LL))(v6);
        if ( **(_DWORD **)(v101 + 8) != 26 )
        {
          HIBYTE(v151) = 1;
          v102 = "expected linked-to symbol";
LABEL_275:
          v148 = v102;
          v103 = *(_QWORD *)(a1 + 8);
          LOBYTE(v151) = 3;
          v104 = sub_ECE0E0(v103, &v148, 0, 0);
          n = 0;
          goto LABEL_276;
        }
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        v144 = 0;
        v145 = 0;
        v106 = sub_ECD690(v101);
        v107 = (*(__int64 (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &v144);
        v108 = *(_QWORD *)(a1 + 8);
        if ( !v107 )
        {
          v109 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v108 + 48LL))(v108);
          v151 = 261;
          v148 = (const char *)v144;
          v149 = v145;
          n = sub_E65280(v109, &v148);
          if ( n && (*(_BYTE *)(n + 9) & 7) == 2 )
          {
            if ( *(_QWORD *)n
              || (*(_BYTE *)(n + 9) & 0x70) == 0x20
              && *(char *)(n + 8) >= 0
              && (*(_BYTE *)(n + 8) |= 8u, v116 = sub_E807D0(*(_QWORD *)(n + 24)), (*(_QWORD *)n = v116) != 0) )
            {
              if ( off_4C5D170 != *(_UNKNOWN **)n )
                goto LABEL_173;
            }
          }
          else
          {
            n = 0;
          }
          v110 = *(_QWORD *)(a1 + 8);
          v151 = 1283;
          v148 = "linked-to symbol is not in a section: ";
          v150[0] = v144;
          v150[1] = v145;
          v104 = sub_ECDA70(v110, v106, &v148, 0, 0);
LABEL_276:
          if ( v104 )
            return 1;
LABEL_173:
          v6 = *(_DWORD **)(a1 + 8);
          if ( !(_BYTE)v27 )
            goto LABEL_178;
          v72 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v6 + 40LL))(v6);
          if ( **(_DWORD **)(v72 + 8) != 26 )
          {
            HIBYTE(v151) = 1;
            v73 = "expected group name";
            goto LABEL_176;
          }
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
          v105 = *(_QWORD *)(a1 + 8);
          if ( **(_DWORD **)(v72 + 8) == 4 )
          {
            v112 = sub_ECD7B0(v105);
            v113 = *(_QWORD *)(a1 + 8);
            v114 = *(_QWORD *)(v112 + 16);
            v115 = *(const char **)(v112 + 8);
            v136 = v114;
            v135 = v115;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v113 + 184LL))(v113);
          }
          else if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v105 + 192LL))(v105, &v135) )
          {
            HIBYTE(v151) = 1;
            v73 = "invalid group name";
LABEL_176:
            v148 = v73;
            v74 = *(_QWORD *)(a1 + 8);
            LOBYTE(v151) = 3;
            LOBYTE(v27) = sub_ECE0E0(v74, &v148, 0, 0);
            if ( !(_BYTE)v27 )
            {
LABEL_177:
              v6 = *(_DWORD **)(a1 + 8);
              goto LABEL_178;
            }
            return 1;
          }
          if ( **(_DWORD **)(v72 + 8) != 26 )
          {
            LOBYTE(v27) = 0;
            goto LABEL_177;
          }
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
          v111 = *(_QWORD *)(a1 + 8);
          v144 = 0;
          v145 = 0;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v111 + 192LL))(v111, &v144) )
          {
            HIBYTE(v151) = 1;
            v73 = "invalid linkage";
          }
          else
          {
            if ( v145 == 6 && !memcmp(v144, "comdat", 6u) )
              goto LABEL_177;
            HIBYTE(v151) = 1;
            v73 = "Linkage must be 'comdat'";
          }
          goto LABEL_176;
        }
        v117 = sub_ECD7B0(v108);
        if ( *(_QWORD *)(v117 + 16) != 1 || **(_BYTE **)(v117 + 8) != 48 )
        {
          HIBYTE(v151) = 1;
          v102 = "invalid linked-to symbol";
          goto LABEL_275;
        }
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
LABEL_172:
        n = 0;
        goto LABEL_173;
      }
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 8) + 192LL))(
              *(_QWORD *)(a1 + 8),
              &s1) )
        goto LABEL_208;
      v87 = *(_QWORD *)(a1 + 8);
      v148 = "expected identifier";
      v151 = 259;
      goto LABEL_207;
    }
LABEL_66:
    v127 = 6;
    goto LABEL_27;
  }
  v7 = v6[6];
  if ( v7 )
    goto LABEL_60;
  v8 = 0;
  v9 = 0;
  while ( 1 )
  {
    v14 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v6 + 40LL))(v6);
    v15 = sub_ECD690(v14);
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26
      || **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      break;
    }
    v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
    v17 = *(_QWORD *)(a1 + 8);
    if ( **(_DWORD **)(v16 + 8) == 3 )
    {
      v22 = sub_ECD7B0(v17);
      v23 = *(_QWORD *)(v22 + 16);
      if ( *(_DWORD *)v22 == 2 )
      {
        v25 = (unsigned int)(v23 + 2);
        v26 = v23 + 2;
      }
      else if ( v23 )
      {
        v24 = v23 - 1;
        if ( !v24 )
          LODWORD(v24) = 1;
        v25 = (unsigned int)(v24 + 1);
        v26 = v24 + 1;
      }
      else
      {
        v25 = 2;
        v26 = 2;
      }
      v126 = v25;
      v7 += v26;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v11 = v126;
      goto LABEL_5;
    }
    v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL))(v17);
    v19 = *(_QWORD *)(a1 + 8);
    if ( **(_DWORD **)(v18 + 8) != 2 )
    {
      v10 = *(_QWORD *)(sub_ECD7B0(v19) + 16);
      v7 += v10;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v11 = (unsigned int)v10;
      goto LABEL_5;
    }
    v20 = sub_ECD7B0(v19);
    v21 = *(_QWORD *)(v20 + 16);
    if ( *(_DWORD *)v20 == 2 )
      goto LABEL_16;
    if ( v21 )
    {
      if ( !--v21 )
        LODWORD(v21) = 1;
      LODWORD(v21) = v21 - 1;
LABEL_16:
      v7 += v21;
      v21 = (unsigned int)v21;
    }
    v125 = v21;
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v11 = v125;
LABEL_5:
    v12 = v11 + v15;
    v9 = (_BYTE *)v128;
    v8 = v7;
    v13 = sub_ECD7B0(*(_QWORD *)(a1 + 8));
    if ( v12 == sub_ECD6A0(v13) )
    {
      v6 = *(_DWORD **)(a1 + 8);
      if ( !v6[6] )
        continue;
    }
    break;
  }
  if ( v7 )
    goto LABEL_25;
  v6 = *(_DWORD **)(a1 + 8);
LABEL_60:
  HIBYTE(v151) = 1;
  v46 = "expected identifier";
LABEL_61:
  v148 = v46;
  LOBYTE(v151) = 3;
  return sub_ECE0E0(v6, &v148, 0, 0);
}
