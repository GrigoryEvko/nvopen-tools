// Function: sub_74A390
// Address: 0x74a390
//
void __fastcall sub_74A390(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 a6)
{
  int v7; // ebx
  __int64 v8; // r14
  __int64 v9; // r12
  char v10; // al
  __int64 v11; // r15
  unsigned int v12; // eax
  unsigned int v13; // ecx
  int v14; // r15d
  char v15; // di
  __int64 v16; // rbx
  unsigned int (__fastcall *v17)(__int64, __int64 *); // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  char v23; // dl
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  char v27; // al
  _BOOL8 v28; // rsi
  __int64 v29; // rdi
  void (__fastcall *v30)(__int64, _BOOL8); // rax
  __int64 (__fastcall *v31)(const char *, __int64); // rdx
  __int64 v32; // r10
  __int64 v33; // rax
  char v34; // dl
  __int64 v35; // rax
  char v36; // dl
  __int64 v37; // rax
  void (__fastcall *v38)(_QWORD, __int64, _QWORD); // rax
  __int64 v39; // rax
  char v40; // dl
  __int64 v41; // rax
  char v42; // dl
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  _BOOL4 v46; // eax
  __int64 v47; // rdi
  void (__fastcall *v48)(_QWORD, __int64, _QWORD); // rax
  int v49; // edx
  __int64 *v50; // rax
  void (__fastcall *v51)(_QWORD, __int64, _QWORD); // rax
  unsigned __int64 v52; // rcx
  void (__fastcall *v53)(__int64, __int64); // rax
  __int64 v54; // rdx
  __int64 v55; // r8
  __int64 v56; // rdi
  __int64 v57; // rcx
  unsigned __int8 v58; // al
  int v59; // edx
  __int64 (__fastcall *v60)(const char *, __int64); // rax
  char v61; // al
  __int64 v62; // rdx
  __int64 (__fastcall *v63)(const char *, __int64); // rax
  unsigned __int64 v64; // rdi
  __int64 v65; // rax
  char v66; // dl
  __int64 v67; // rax
  char v68; // dl
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rdi
  __int64 (__fastcall *v72)(const char *, __int64); // rdx
  const char *v73; // rdi
  char v74; // al
  char *v75; // rdi
  __int64 *v76; // r8
  __int64 v77; // r8
  void (__fastcall *v78)(__int64, __int64); // rax
  __int64 v79; // rdx
  unsigned __int64 v80; // rax
  __int64 v81; // rdx
  __int64 *v82; // rdi
  void (__fastcall *v83)(__int64 *, __int64); // rax
  __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 v87; // rdi
  char v88; // al
  char *v89; // rdi
  char v90; // dl
  __int64 (__fastcall *v91)(const char *, __int64); // rax
  char *v92; // rdi
  const char *v93; // rax
  int v94; // [rsp+0h] [rbp-B0h]
  unsigned int v96; // [rsp+8h] [rbp-A8h]
  int v98; // [rsp+10h] [rbp-A0h]
  __int64 v99; // [rsp+10h] [rbp-A0h]
  __int64 v100; // [rsp+18h] [rbp-98h]
  __int64 v102; // [rsp+20h] [rbp-90h]
  __int64 v103; // [rsp+20h] [rbp-90h]
  __int64 *v104; // [rsp+20h] [rbp-90h]
  __int64 (__fastcall *v105)(const char *, __int64); // [rsp+20h] [rbp-90h]
  __int64 v106; // [rsp+28h] [rbp-88h] BYREF
  char v107; // [rsp+33h] [rbp-7Dh] BYREF
  unsigned int v108; // [rsp+34h] [rbp-7Ch] BYREF
  __int64 v109; // [rsp+38h] [rbp-78h] BYREF
  _DWORD v110[28]; // [rsp+40h] [rbp-70h] BYREF

  v7 = a5 & 1;
  v106 = a1;
  v94 = a2;
  v98 = a5 & 4;
  v108 = 0;
  v109 = 0;
  if ( !a1 )
  {
    (*(void (__fastcall **)(const char *, __int64))a6)("<something>", a6);
    return;
  }
  v8 = a1;
  v100 = 0;
  if ( HIDWORD(qword_4F077B4) )
  {
    a2 = (__int64)&v107;
    v25 = sub_746E90(a1, &v107);
    a1 = v106;
    v100 = v25;
  }
  v9 = v8;
  v96 = a5 & 0xFFFFFFFE;
  v10 = *(_BYTE *)(a1 + 140);
  if ( v10 == 12 )
  {
    while ( 1 )
    {
      v11 = v106;
      if ( (unsigned int)sub_8D2B80(a1) && *(char *)(v106 + 185) < 0 )
      {
LABEL_72:
        v10 = *(_BYTE *)(v11 + 140);
        a1 = v11;
        goto LABEL_18;
      }
      if ( v109 )
        goto LABEL_10;
      if ( !*(_QWORD *)(v106 + 8) )
        break;
      if ( (*(_BYTE *)(v106 + 89) & 1) != 0 && *(_BYTE *)(a6 + 139) )
        goto LABEL_38;
      if ( !v7 || (*(_BYTE *)(v106 + 140) & 0xFB) != 8 || (a2 = dword_4F077C4 != 2, (sub_8D4C10(v106, a2) & 1) == 0) )
      {
        if ( !*(_BYTE *)(a6 + 138) )
        {
          v17 = *(unsigned int (__fastcall **)(__int64, __int64 *))(a6 + 96);
          if ( !v17 || (a2 = (__int64)&v109, !v17(v106, &v109)) )
          {
            v11 = v106;
            a2 = a6;
            if ( !sub_746100(v106, a6) )
              goto LABEL_72;
LABEL_38:
            a1 = *(_QWORD *)(v11 + 160);
            goto LABEL_15;
          }
        }
      }
      v11 = v106;
      a1 = *(_QWORD *)(v106 + 160);
LABEL_15:
      v10 = *(_BYTE *)(a1 + 140);
      if ( v10 == 6 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(a1 + 160) + 140LL) == 1 && !*(_BYTE *)(a6 + 141) )
        {
          v18 = *(_QWORD *)(v11 + 176);
          if ( v18 )
          {
            if ( *(_BYTE *)(v18 + 140) == 19 )
            {
              v10 = 19;
              a1 = *(_QWORD *)(v11 + 176);
            }
          }
        }
        v106 = a1;
        goto LABEL_18;
      }
      v106 = a1;
      if ( v10 != 12 )
        goto LABEL_18;
    }
    if ( !sub_5D7700() )
    {
      a2 = a6;
      if ( (unsigned int)sub_746C80(v106, a6) )
      {
        a1 = v106;
        v10 = *(_BYTE *)(v106 + 140);
        goto LABEL_18;
      }
    }
    v11 = v106;
LABEL_10:
    v12 = v108 | *(_BYTE *)(v11 + 185) & 0x7F;
    v108 = v12;
    if ( v7 && (v12 & 1) != 0 )
    {
      v7 = 0;
      v108 = v12 & 0xFFFFFFFE;
    }
    a1 = *(_QWORD *)(v11 + 160);
    if ( *(_BYTE *)(v11 + 184) == 8 )
      v9 = *(_QWORD *)(v11 + 160);
    goto LABEL_15;
  }
LABEL_18:
  v13 = v108 | a4;
  v108 |= a4;
  if ( v98 && (v13 & 4) != 0 )
    v108 = v13 & 0xFFFFFFFB;
  switch ( v10 )
  {
    case 6:
      v19 = *(_QWORD *)(a1 + 160);
      if ( *(_BYTE *)(a6 + 153) && *(_BYTE *)(v19 + 140) == 12 )
      {
        do
        {
          if ( !*(_QWORD *)(v19 + 8) )
            break;
          v20 = v19;
          do
          {
            v20 = *(_QWORD *)(v20 + 160);
            v21 = *(_BYTE *)(v20 + 140);
          }
          while ( v21 == 12 );
          if ( v21 == 21 )
            break;
          v22 = v19;
          do
          {
            v22 = *(_QWORD *)(v22 + 160);
            v23 = *(_BYTE *)(v22 + 140);
          }
          while ( v23 == 12 );
          if ( !v23 )
            break;
          v24 = *(_QWORD *)(v19 + 40);
          if ( v24 )
          {
            if ( *(_BYTE *)(v24 + 28) == 3 && **(_QWORD ***)(v24 + 32) == qword_4D049B8 )
              break;
          }
          v19 = *(_QWORD *)(v19 + 160);
        }
        while ( *(_BYTE *)(v19 + 140) == 12 );
      }
      sub_74A390(v19, 1, 1, 0, v96, a6);
      if ( (*(_BYTE *)(v106 + 168) & 1) != 0
        && (!*(_BYTE *)(a6 + 136) || !*(_BYTE *)(a6 + 141))
        && !(unsigned int)sub_5D7F10() )
      {
        v31 = *(__int64 (__fastcall **)(const char *, __int64))a6;
        if ( (*(_BYTE *)(v106 + 168) & 2) != 0 )
          v31("&&", a6);
        else
          v31("&", a6);
        goto LABEL_63;
      }
LABEL_62:
      (*(void (__fastcall **)(char *, __int64))a6)("*", a6);
LABEL_63:
      if ( v108 )
        sub_746940(v108, -1, a3, a6);
      if ( v8 != v9 )
      {
        if ( *(_BYTE *)(a6 + 136) )
          (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
        sub_74A2C0(v8, v9, a6);
      }
      goto LABEL_69;
    case 13:
      v26 = *(_QWORD *)(a1 + 168);
      if ( *(_BYTE *)(a6 + 153) )
      {
        while ( *(_BYTE *)(v26 + 140) == 12 )
        {
          if ( !*(_QWORD *)(v26 + 8) )
            break;
          v39 = v26;
          do
          {
            v39 = *(_QWORD *)(v39 + 160);
            v40 = *(_BYTE *)(v39 + 140);
          }
          while ( v40 == 12 );
          if ( v40 == 21 )
            break;
          v41 = v26;
          do
          {
            v41 = *(_QWORD *)(v41 + 160);
            v42 = *(_BYTE *)(v41 + 140);
          }
          while ( v42 == 12 );
          if ( !v42 )
            break;
          v43 = *(_QWORD *)(v26 + 40);
          if ( v43 )
          {
            if ( *(_BYTE *)(v43 + 28) == 3 && **(_QWORD ***)(v43 + 32) == qword_4D049B8 )
              break;
          }
          v26 = *(_QWORD *)(v26 + 160);
        }
      }
      sub_74A390(v26, 1, 1, 0, v96, a6);
      v27 = *(_BYTE *)(v26 + 140);
      if ( *(_BYTE *)(a6 + 136) && v27 != 7 && !*(_BYTE *)(a6 + 154) )
      {
        (*(void (__fastcall **)(char *, __int64))a6)("(", a6);
        v27 = *(_BYTE *)(v26 + 140);
      }
      v28 = v27 != 7;
      v29 = *(_QWORD *)(v106 + 160);
      v30 = *(void (__fastcall **)(__int64, _BOOL8))(a6 + 40);
      if ( v30 )
        v30(v29, v28);
      else
        sub_74C3E0(v29, a6);
      goto LABEL_62;
    case 7:
      v44 = *(_QWORD *)(a1 + 168);
      v45 = *(_QWORD *)(v44 + 8);
      v46 = (*(_BYTE *)(v44 + 17) & 4) != 0;
      if ( v45 && *(_BYTE *)(v45 + 174) == 7 )
      {
        v49 = 1;
      }
      else
      {
        if ( (*(_BYTE *)(v44 + 16) & 8) != 0 )
        {
          v49 = 0;
          if ( !*(_BYTE *)(a6 + 136) )
            goto LABEL_129;
          goto LABEL_128;
        }
        if ( (*(_BYTE *)(v44 + 17) & 4) == 0 )
        {
LABEL_119:
          v47 = *(_QWORD *)(a1 + 160);
          if ( *(_BYTE *)(a6 + 153) )
          {
            while ( *(_BYTE *)(v47 + 140) == 12 )
            {
              if ( !*(_QWORD *)(v47 + 8) )
                break;
              v65 = v47;
              do
              {
                v65 = *(_QWORD *)(v65 + 160);
                v66 = *(_BYTE *)(v65 + 140);
              }
              while ( v66 == 12 );
              if ( v66 == 21 )
                break;
              v67 = v47;
              do
              {
                v67 = *(_QWORD *)(v67 + 160);
                v68 = *(_BYTE *)(v67 + 140);
              }
              while ( v68 == 12 );
              if ( !v68 )
                break;
              v69 = *(_QWORD *)(v47 + 40);
              if ( v69 )
              {
                if ( *(_BYTE *)(v69 + 28) == 3 && **(_QWORD ***)(v69 + 32) == qword_4D049B8 )
                  break;
              }
              v47 = *(_QWORD *)(v47 + 160);
            }
          }
          sub_74A390(v47, 0, 1, 0, v96, a6);
          goto LABEL_122;
        }
        v49 = 0;
      }
      if ( !*(_BYTE *)(a6 + 136) )
      {
LABEL_122:
        if ( v94 )
        {
          (*(void (__fastcall **)(char *, __int64))a6)("(", a6);
          v48 = *(void (__fastcall **)(_QWORD, __int64, _QWORD))(a6 + 88);
          if ( v48 )
            v48(*(_QWORD *)(v106 + 104), 11, 0);
        }
        if ( v108 )
          sub_746940(v108, -1, a3, a6);
        goto LABEL_69;
      }
LABEL_128:
      if ( *(_BYTE *)(a6 + 141) )
        goto LABEL_119;
LABEL_129:
      if ( !(v49 | a5 & 2 | v46) )
        (*(void (__fastcall **)(const char *, __int64))a6)("auto ", a6);
      goto LABEL_122;
    case 8:
      v32 = *(_QWORD *)(a1 + 160);
      if ( *(_BYTE *)(a6 + 153) && *(_BYTE *)(v32 + 140) == 12 )
      {
        do
        {
          if ( !*(_QWORD *)(v32 + 8) )
            break;
          v33 = v32;
          do
          {
            v33 = *(_QWORD *)(v33 + 160);
            v34 = *(_BYTE *)(v33 + 140);
          }
          while ( v34 == 12 );
          if ( v34 == 21 )
            break;
          v35 = v32;
          do
          {
            v35 = *(_QWORD *)(v35 + 160);
            v36 = *(_BYTE *)(v35 + 140);
          }
          while ( v36 == 12 );
          if ( !v36 )
            break;
          v37 = *(_QWORD *)(v32 + 40);
          if ( v37 )
          {
            if ( *(_BYTE *)(v37 + 28) == 3 && **(_QWORD ***)(v37 + 32) == qword_4D049B8 )
              break;
          }
          v32 = *(_QWORD *)(v32 + 160);
        }
        while ( *(_BYTE *)(v32 + 140) == 12 );
      }
      a2 = (__int64)&v108;
      v99 = v32;
      if ( !(unsigned int)sub_745B20(&v106, &v108, v7, a6) )
      {
        sub_74A390(v99, 0, 1, 0, v7 | v96, a6);
        if ( v94 )
        {
          (*(void (__fastcall **)(char *, __int64))a6)("(", a6);
          v38 = *(void (__fastcall **)(_QWORD, __int64, _QWORD))(a6 + 88);
          if ( v38 )
            v38(*(_QWORD *)(v106 + 104), 11, 0);
        }
        goto LABEL_69;
      }
      break;
  }
  v14 = a5 & 2;
  if ( (a5 & 2) == 0 )
  {
    v15 = v108;
    if ( v108 )
    {
      if ( (v108 & 8) != 0 )
      {
        v108 &= ~8u;
        v14 = 1;
        sub_746940(v15 & 0xF7, -1, 1, a6);
        a2 = a6;
        (*(void (__fastcall **)(const char *, __int64))a6)("_Atomic(", a6);
      }
      else
      {
        a2 = -1;
        sub_746940(v108, -1, 1, a6);
      }
    }
    v16 = v106;
    if ( *(_BYTE *)(a6 + 141) && sub_5D76E0() && (unsigned int)sub_8D2FF0(v16, a2) )
    {
      if ( (unsigned int)sub_8D3030(v16) )
        (*(void (__fastcall **)(const char *, __int64))a6)("__surface_type__", a6);
      else
        (*(void (__fastcall **)(const char *, __int64))a6)("__texture_type__", a6);
      goto LABEL_137;
    }
    switch ( *(_BYTE *)(v16 + 140) )
    {
      case 0:
        (*(void (__fastcall **)(const char *, __int64))a6)("<error-type>", a6);
        goto LABEL_137;
      case 1:
        (*(void (__fastcall **)(char *, __int64))a6)("void", a6);
        goto LABEL_137;
      case 2:
        v61 = *(_BYTE *)(v16 + 161);
        if ( (v61 & 8) != 0 )
        {
          if ( !*(_BYTE *)(a6 + 136) || !*(_BYTE *)(a6 + 141) || !sub_5D76E0() )
            goto LABEL_136;
          v61 = *(_BYTE *)(v16 + 161);
          if ( !(*(_BYTE *)(a6 + 137) | v61 & 4) && (**(_BYTE **)(v16 + 176) & 1) != 0 )
          {
            v62 = *(_QWORD *)(v16 + 168);
            if ( (v61 & 0x10) != 0 )
              v62 = *(_QWORD *)(v62 + 96);
            if ( v62 )
              goto LABEL_136;
          }
        }
        if ( (v61 & 0x40) == 0 )
          goto LABEL_252;
        if ( *(_BYTE *)(a6 + 136) && *(_BYTE *)(a6 + 141) || sub_5D7700() )
        {
          (*(void (__fastcall **)(char *, __int64))a6)("wchar_t", a6);
          goto LABEL_137;
        }
        v61 = *(_BYTE *)(v16 + 161);
LABEL_252:
        if ( v61 < 0 )
        {
          if ( !*(_BYTE *)(a6 + 136) || !*(_BYTE *)(a6 + 141) )
          {
            (*(void (__fastcall **)(char *, __int64))a6)("char8_t", a6);
            goto LABEL_137;
          }
          v74 = *(_BYTE *)(v16 + 162);
          if ( (v74 & 1) == 0 )
          {
            if ( (v74 & 2) != 0 )
              goto LABEL_315;
            goto LABEL_318;
          }
        }
        else
        {
          v74 = *(_BYTE *)(v16 + 162);
          if ( (v74 & 1) == 0 )
            goto LABEL_308;
          if ( !*(_BYTE *)(a6 + 136) )
          {
LABEL_255:
            (*(void (__fastcall **)(char *, __int64))a6)("char16_t", a6);
            goto LABEL_137;
          }
          if ( !*(_BYTE *)(a6 + 141) )
          {
            if ( !dword_4F068D0 )
              goto LABEL_255;
            goto LABEL_305;
          }
        }
        if ( sub_5D7700() )
        {
          if ( !dword_4F068D0 || !*(_BYTE *)(a6 + 136) )
            goto LABEL_255;
LABEL_305:
          (*(void (__fastcall **)(const char *, __int64))a6)("__char16_t", a6);
          goto LABEL_137;
        }
        v74 = *(_BYTE *)(v16 + 162);
LABEL_308:
        if ( (v74 & 2) != 0 )
        {
          if ( !*(_BYTE *)(a6 + 136) )
          {
LABEL_310:
            (*(void (__fastcall **)(char *, __int64))a6)("char32_t", a6);
            goto LABEL_137;
          }
LABEL_315:
          if ( *(_BYTE *)(a6 + 141) )
          {
            if ( !sub_5D7700() )
            {
              v74 = *(_BYTE *)(v16 + 162);
              goto LABEL_318;
            }
            if ( !dword_4F068D0 || !*(_BYTE *)(a6 + 136) )
              goto LABEL_310;
          }
          else if ( !dword_4F068D0 )
          {
            goto LABEL_310;
          }
          (*(void (__fastcall **)(const char *, __int64))a6)("__char32_t", a6);
          goto LABEL_137;
        }
LABEL_318:
        if ( (v74 & 4) != 0 )
        {
          if ( sub_5D7700() )
          {
            (*(void (__fastcall **)(char *, __int64))a6)("bool", a6);
            goto LABEL_137;
          }
          if ( (*(_BYTE *)(v16 + 162) & 4) != 0 )
          {
            if ( sub_5D76E0() )
            {
              (*(void (__fastcall **)(const char *, __int64))a6)("__nv_bool", a6);
              goto LABEL_137;
            }
            if ( (*(_BYTE *)(v16 + 162) & 4) != 0 )
            {
              v90 = *(_BYTE *)(a6 + 140);
              if ( !*(_BYTE *)(a6 + 136) || !*(_BYTE *)(a6 + 141) )
              {
                v92 = "_Bool";
                v91 = *(__int64 (__fastcall **)(const char *, __int64))a6;
                if ( !v90 )
                  v92 = "bool";
                goto LABEL_350;
              }
              if ( v90 )
              {
                v91 = *(__int64 (__fastcall **)(const char *, __int64))a6;
                v92 = "_Bool";
LABEL_350:
                v91(v92, a6);
                goto LABEL_137;
              }
            }
          }
        }
        if ( (*(_BYTE *)(v16 + 161) & 1) != 0 )
        {
          if ( *(_BYTE *)(a6 + 137) )
          {
            v88 = *(_BYTE *)(v16 + 160);
LABEL_330:
            v89 = "char";
            if ( v88 == 1 )
              goto LABEL_326;
            if ( v88 == 2 )
            {
              if ( !unk_4F072A8 )
                goto LABEL_326;
              goto LABEL_333;
            }
LABEL_323:
            if ( v88 == 6 && *(_BYTE *)(a6 + 136) )
            {
              v89 = "unsigned";
LABEL_326:
              (*(void (__fastcall **)(char *, __int64))a6)(v89, a6);
              goto LABEL_137;
            }
LABEL_333:
            v89 = sub_7464F0(v16, *(unsigned __int8 *)(a6 + 136));
            goto LABEL_326;
          }
          (*(void (__fastcall **)(char *, __int64))a6)("signed ", a6);
        }
        v88 = *(_BYTE *)(v16 + 160);
        if ( !*(_BYTE *)(a6 + 137) )
          goto LABEL_323;
        goto LABEL_330;
      case 3:
        sub_746720(*(_BYTE *)(v16 + 160), a6);
        goto LABEL_137;
      case 4:
        sub_746720(*(_BYTE *)(v16 + 160), a6);
        (*(void (__fastcall **)(const char *, __int64))a6)(" _Imaginary", a6);
        goto LABEL_137;
      case 5:
        sub_746720(*(_BYTE *)(v16 + 160), a6);
        if ( *(_BYTE *)(a6 + 136) )
          v59 = dword_4F068C4;
        else
          v59 = HIDWORD(qword_4F077B4);
        v60 = *(__int64 (__fastcall **)(const char *, __int64))a6;
        if ( v59 )
          v60(" __complex__", a6);
        else
          v60(" _Complex", a6);
        goto LABEL_137;
      case 9:
      case 0xA:
      case 0xB:
LABEL_136:
        sub_74D000(v16, a6);
        goto LABEL_137;
      case 0xC:
        if ( sub_5D7700() || !(unsigned int)sub_746C80(v16, a6) )
        {
          v52 = *(unsigned __int8 *)(v16 + 184);
          if ( (_BYTE)v52 == 1 )
          {
            if ( *(_BYTE *)(a6 + 136) )
            {
              v53 = *(void (__fastcall **)(__int64, __int64))(a6 + 24);
              if ( v53 )
                goto LABEL_165;
            }
            goto LABEL_257;
          }
        }
        else
        {
          v52 = *(unsigned __int8 *)(v16 + 184);
          if ( *(_BYTE *)(a6 + 136) )
          {
            v53 = *(void (__fastcall **)(__int64, __int64))(a6 + 24);
            if ( v53 )
            {
              if ( (unsigned __int8)v52 > 0xAu || ((0x71DuLL >> v52) & 1) == 0 || (_BYTE)v52 == 1 )
                goto LABEL_165;
LABEL_161:
              v54 = 1821;
              if ( _bittest64(&v54, v52) )
              {
                if ( (unsigned __int8)(v52 - 6) > 1u )
                {
                  switch ( (_BYTE)v52 )
                  {
                    case 2:
                      v87 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 24LL);
                      if ( v87 )
                      {
                        sub_747C50(v87, a6);
                        (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
                      }
                      (*(void (__fastcall **)(const char *, __int64))a6)("decltype(auto)", a6);
                      break;
                    case 3:
                      v86 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 24LL);
                      if ( v86 )
                      {
                        sub_747C50(v86, a6);
                        (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
                      }
                      (*(void (__fastcall **)(char *, __int64))a6)("auto", a6);
                      break;
                    case 4:
                      sub_74C550(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16 + 160) + 168LL) + 160LL), 59, a6);
                      break;
                    default:
                      sub_74C550(v16, 6, a6);
                      break;
                  }
                  goto LABEL_137;
                }
              }
              else if ( (unsigned __int8)(v52 - 6) > 1u )
              {
                goto LABEL_163;
              }
              if ( *(_BYTE *)(a6 + 136) )
              {
                v53 = *(void (__fastcall **)(__int64, __int64))(a6 + 24);
                if ( v53 )
                  goto LABEL_165;
              }
              (*(void (__fastcall **)(const char *, __int64))a6)("__typeof__(", a6);
              if ( *(_BYTE *)(v16 + 184) == 7 )
              {
                sub_74B930(*(_QWORD *)(v16 + 160), a6);
                goto LABEL_169;
              }
              v82 = sub_746BE0(v16);
              if ( v82 )
              {
                v83 = *(void (__fastcall **)(__int64 *, __int64))(a6 + 72);
                if ( v83 )
                  v83(v82, 1);
                else
                  sub_747C50((__int64)v82, a6);
                goto LABEL_169;
              }
LABEL_275:
              (*(void (__fastcall **)(const char *, __int64))a6)("<expr>", a6);
              goto LABEL_169;
            }
          }
          if ( (_BYTE)v52 == 1 )
          {
LABEL_257:
            v75 = "decltype(";
            v76 = sub_746BE0(v16);
            if ( *(_BYTE *)(a6 + 136) && dword_4F068C4 )
              v75 = "__decltype(";
            v104 = v76;
            (*(void (__fastcall **)(char *, __int64))a6)(v75, a6);
            v77 = (__int64)v104;
            if ( v104 )
            {
              if ( (*(_BYTE *)(v16 + 186) & 2) == 0 )
              {
                (*(void (__fastcall **)(char *, __int64))a6)("(", a6);
                v77 = (__int64)v104;
              }
              v78 = *(void (__fastcall **)(__int64, __int64))(a6 + 72);
              if ( v78 )
                v78(v77, 1);
              else
                sub_747C50(v77, a6);
              if ( (*(_BYTE *)(v16 + 186) & 2) == 0 )
                (*(void (__fastcall **)(char *, __int64))a6)(")", a6);
              goto LABEL_169;
            }
            goto LABEL_275;
          }
        }
        if ( (unsigned __int8)v52 > 0xAu )
        {
          if ( (unsigned __int8)(v52 - 11) <= 1u )
          {
            if ( *(_BYTE *)(a6 + 136) )
            {
              v53 = *(void (__fastcall **)(__int64, __int64))(a6 + 24);
              if ( v53 )
                goto LABEL_165;
            }
            v73 = "__direct_bases(";
            if ( (_BYTE)v52 != 12 )
              v73 = "__bases(";
            (*(void (__fastcall **)(const char *, __int64))a6)(v73, a6);
LABEL_246:
            sub_74B930(*(_QWORD *)(*(_QWORD *)(v16 + 168) + 40LL), a6);
LABEL_169:
            (*(void (__fastcall **)(char *, __int64))a6)(")", a6);
LABEL_137:
            if ( v14 )
              (*(void (__fastcall **)(char *, __int64))a6)(")", a6);
            if ( v8 != v9 )
            {
              if ( *(_BYTE *)(a6 + 136) )
                (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
              sub_74A2C0(v8, v9, a6);
            }
            v50 = *(__int64 **)(v8 + 104);
            if ( v50 )
            {
              while ( *((_BYTE *)v50 + 10) != 11 )
              {
                v50 = (__int64 *)*v50;
                if ( !v50 )
                  goto LABEL_149;
              }
              (*(void (__fastcall **)(char *, __int64))a6)(" (", a6);
              v51 = *(void (__fastcall **)(_QWORD, __int64, _QWORD))(a6 + 88);
              if ( v51 )
                v51(*(_QWORD *)(v8 + 104), 11, 0);
            }
LABEL_149:
            if ( a3 )
              (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
            break;
          }
LABEL_163:
          if ( *(_BYTE *)(a6 + 136) )
          {
            v53 = *(void (__fastcall **)(__int64, __int64))(a6 + 24);
            if ( v53 )
            {
LABEL_165:
              v53(v16, 6);
              goto LABEL_137;
            }
          }
          v105 = *(__int64 (__fastcall **)(const char *, __int64))a6;
          v93 = sub_746810(v52);
          v105(v93, a6);
          (*(void (__fastcall **)(char *, __int64))a6)("(", a6);
          goto LABEL_246;
        }
        goto LABEL_161;
      case 0xE:
        if ( (unsigned int)sub_8D3EA0(v16) || (*(_BYTE *)(v16 + 161) & 4) != 0 )
        {
          v70 = *(_QWORD *)(v16 + 168);
          v71 = *(_QWORD *)(v70 + 32);
          if ( v71 )
          {
            sub_747C50(v71, a6);
            (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
            v70 = *(_QWORD *)(v16 + 168);
          }
          v72 = *(__int64 (__fastcall **)(const char *, __int64))a6;
          if ( *(_QWORD *)(v70 + 24) == 0xFFFFFFFF00000002LL )
            v72("decltype(auto)", a6);
          else
            v72("auto", a6);
        }
        else
        {
          v58 = 6;
          if ( *(_BYTE *)(v16 + 140) == 14 && !*(_BYTE *)(v16 + 160) )
          {
            v84 = *(_QWORD *)(v16 + 168);
            if ( *(int *)(v84 + 28) > 0 )
            {
              v85 = sub_746B90((_DWORD *)(v84 + 24));
              if ( v85 )
                v16 = v85;
              v58 = v85 == 0 ? 6 : 64;
            }
          }
          if ( (*(_BYTE *)(v16 + 90) & 0x30) == 0x10 )
            sub_74C010(v16, v58, a6);
          else
            sub_74C550(v16, v58, a6);
        }
        goto LABEL_137;
      case 0xF:
        if ( unk_4F066B0 )
        {
          v102 = *(_QWORD *)(v16 + 160);
          (*(void (__fastcall **)(const char *, __int64))a6)("__edg_vector_type__(", a6);
          sub_74B930(v102, a6);
          (*(void (__fastcall **)(char *, __int64))a6)(", ", a6);
          v56 = *(_QWORD *)(v16 + 168);
          v57 = v102;
          if ( v56 )
          {
            sub_748000(v56, 0, a6, v102, v55);
          }
          else
          {
            if ( *(_BYTE *)(v102 + 140) == 12 )
            {
              v79 = v102;
              do
                v79 = *(_QWORD *)(v79 + 160);
              while ( *(_BYTE *)(v79 + 140) == 12 );
              v57 = v79;
            }
            v80 = *(_QWORD *)(v16 + 128) / *(_QWORD *)(v57 + 128);
            v81 = *(_QWORD *)(v16 + 128) % *(_QWORD *)(v57 + 128);
            if ( v80 > 9 )
              sub_622470(v80, v110);
            else
              LOWORD(v110[0]) = (unsigned __int8)(v80 + 48);
            (*(void (__fastcall **)(_DWORD *, __int64, __int64))a6)(v110, a6, v81);
          }
          goto LABEL_169;
        }
        if ( !*(_BYTE *)(a6 + 151) && !*(_QWORD *)(a6 + 88) )
        {
          v110[0] = 0;
          sub_749E60(v16, v110, (__int64 (__fastcall **)(const char *, _QWORD))a6);
          (*(void (__fastcall **)(char *, __int64))a6)(" ", a6);
        }
        sub_74B930(*(_QWORD *)(v16 + 160), a6);
        goto LABEL_137;
      case 0x10:
        v103 = *(_QWORD *)(v16 + 160);
        (*(void (__fastcall **)(const char *, __int64))a6)("__edg_scalable_vector_type__(", a6);
        sub_74B930(v103, a6);
        (*(void (__fastcall **)(char *, __int64))a6)(", ", a6);
        v64 = *(unsigned __int8 *)(v16 + 168);
        if ( (unsigned __int8)v64 > 9u )
          sub_622470(v64, v110);
        else
          LOWORD(v110[0]) = (unsigned __int8)(v64 + 48);
        (*(void (__fastcall **)(_DWORD *, __int64))a6)(v110, a6);
        (*(void (__fastcall **)(char *, __int64))a6)(")", a6);
        goto LABEL_137;
      case 0x11:
        (*(void (__fastcall **)(char *, __int64))a6)("__SVCount_t", a6);
        goto LABEL_137;
      case 0x12:
        (*(void (__fastcall **)(char *, __int64))a6)("__mfp8", a6);
        goto LABEL_137;
      case 0x13:
        if ( (unsigned int)sub_8D26D0(v16) )
          (*(void (__fastcall **)(char *, __int64))a6)("std::nullptr_t", a6);
        else
          (*(void (__fastcall **)(const char *, __int64))a6)("decltype(nullptr)", a6);
        goto LABEL_137;
      case 0x14:
        v63 = *(__int64 (__fastcall **)(const char *, __int64))a6;
        if ( *(_BYTE *)(a6 + 136) )
          v63("decltype(^::)", a6);
        else
          v63("std::meta::info", a6);
        goto LABEL_137;
      case 0x15:
        (*(void (__fastcall **)(const char *, __int64))a6)("<unknown-type>", a6);
        goto LABEL_137;
      default:
        sub_721090();
    }
  }
LABEL_69:
  if ( v100 )
    *(_BYTE *)(v100 + 160) = v107;
}
