// Function: sub_392C2E0
// Address: 0x392c2e0
//
__int64 __fastcall sub_392C2E0(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v4; // ebx
  char v6; // si
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  size_t v11; // rax
  size_t v12; // rax
  __int64 v13; // rdx
  unsigned __int64 *v14; // rax
  __m128i si128; // xmm0
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  _BYTE *v19; // r15
  char *v20; // rcx
  char *v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rdx
  char v29; // r15
  __int64 v30; // rbx
  bool v31; // r9
  unsigned __int64 v32; // rcx
  size_t v33; // rax
  unsigned __int64 v34; // r14
  int v35; // eax
  int v36; // eax
  unsigned int v37; // edx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  _BYTE *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  _BYTE *v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  _BYTE *v58; // rax
  __int64 v59; // rdx
  char *v60; // rax
  __int64 v61; // rcx
  char v62; // dl
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // eax
  bool v69; // r8
  bool v70; // r8
  int v71; // eax
  int v72; // eax
  unsigned __int64 v73; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v74; // [rsp+8h] [rbp-A8h]
  bool v75; // [rsp+18h] [rbp-98h]
  unsigned __int64 v76; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int64 *v77; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v78; // [rsp+38h] [rbp-78h]
  unsigned __int64 v79; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v80; // [rsp+48h] [rbp-68h]
  unsigned int v81; // [rsp+50h] [rbp-60h]
  int v82; // [rsp+58h] [rbp-58h] BYREF
  __int64 v83; // [rsp+60h] [rbp-50h]
  __int64 v84; // [rsp+68h] [rbp-48h]
  unsigned __int64 v85; // [rsp+70h] [rbp-40h]
  unsigned int v86; // [rsp+78h] [rbp-38h]

  *(_QWORD *)(a2 + 104) = *(_QWORD *)(a2 + 144);
  v3 = sub_392A7D0((_QWORD *)a2);
  v4 = v3;
  if ( *(_BYTE *)(a2 + 171) != 1 && v3 == 35 && *(_BYTE *)(a2 + 169) )
  {
    v7 = *(_QWORD *)a2;
    v78 = 0;
    v79 = 0;
    v81 = 1;
    v80 = 0;
    v83 = 0;
    v84 = 0;
    v86 = 1;
    v85 = 0;
    v8 = (*(__int64 (__fastcall **)(__int64, unsigned __int64 **, __int64, __int64))(v7 + 32))(a2, &v77, 2, 1);
    if ( *(_BYTE *)(a2 + 168) && v8 == 2 && (_DWORD)v77 == 4 && v82 == 3 )
    {
      *(_QWORD *)(a2 + 144) = *(_QWORD *)(a2 + 104);
      v16 = sub_392BEB0((_QWORD *)a2);
      *(_BYTE *)(a2 + 114) = 0;
      v18 = v17;
      v19 = v16;
      sub_392C040(a2 + 8, *(_QWORD *)(a2 + 8), (unsigned __int64)&v82);
      *(_BYTE *)(a2 + 114) = 0;
      sub_392C040(a2 + 8, *(_QWORD *)(a2 + 8), (unsigned __int64)&v77);
      *(_DWORD *)a1 = 8;
      *(_QWORD *)(a1 + 8) = v19;
      *(_QWORD *)(a1 + 16) = v18;
      *(_DWORD *)(a1 + 32) = 64;
      *(_QWORD *)(a1 + 24) = 0;
    }
    else
    {
      sub_392AD60(a1, a2);
    }
    if ( v86 > 0x40 && v85 )
      j_j___libc_free_0_0(v85);
    if ( v81 > 0x40 && v80 )
      j_j___libc_free_0_0(v80);
    return a1;
  }
  if ( sub_392BF20(a2, *(const char **)(a2 + 104)) )
  {
    sub_392AD60(a1, a2);
    return a1;
  }
  if ( sub_392BF70(a2, *(const char **)(a2 + 104)) )
  {
    v10 = *(_QWORD *)(a2 + 136);
    v11 = strlen(*(const char **)(v10 + 40));
    *(_WORD *)(a2 + 168) = 257;
    *(_QWORD *)(a2 + 144) += v11 - 1;
    v12 = strlen(*(const char **)(v10 + 40));
    v13 = *(_QWORD *)(a2 + 104);
    *(_DWORD *)a1 = 9;
    *(_QWORD *)(a1 + 16) = v12;
    *(_QWORD *)(a1 + 8) = v13;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  v6 = *(_BYTE *)(a2 + 169);
  if ( v4 != -1 )
  {
    *(_WORD *)(a2 + 168) = 0;
    switch ( v4 )
    {
      case 0:
      case 9:
      case 32:
        v20 = *(char **)(a2 + 144);
        *(_BYTE *)(a2 + 169) = v6;
        if ( *v20 == 32 || *v20 == 9 )
        {
          v21 = v20 + 1;
          do
          {
            do
            {
              *(_QWORD *)(a2 + 144) = v21;
              v22 = *v21;
              v20 = v21++;
            }
            while ( v22 == 32 );
          }
          while ( v22 == 9 );
        }
        if ( *(_BYTE *)(a2 + 112) )
        {
          (**(void (__fastcall ***)(__int64, __int64))a2)(a1, a2);
        }
        else
        {
          v67 = *(_QWORD *)(a2 + 104);
          *(_DWORD *)a1 = 11;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 8) = v67;
          *(_QWORD *)(a1 + 16) = &v20[-v67];
          *(_QWORD *)(a1 + 24) = 0;
        }
        return a1;
      case 10:
        *(_WORD *)(a2 + 168) = 257;
        v46 = *(_QWORD *)(a2 + 104);
        goto LABEL_71;
      case 13:
        v44 = *(_BYTE **)(a2 + 144);
        *(_WORD *)(a2 + 168) = 257;
        if ( v44 != (_BYTE *)(*(_QWORD *)(a2 + 152) + *(_QWORD *)(a2 + 160)) && *v44 == 10 )
          *(_QWORD *)(a2 + 144) = ++v44;
        v45 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 9;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 8) = v45;
        *(_QWORD *)(a1 + 16) = &v44[-v45];
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 33:
        v41 = *(_BYTE **)(a2 + 144);
        v42 = *(_QWORD *)(a2 + 104);
        if ( *v41 == 61 )
        {
          *(_QWORD *)(a2 + 144) = v41 + 1;
          *(_DWORD *)a1 = 35;
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)(a1 + 16) = 2;
        }
        else
        {
          *(_DWORD *)a1 = 34;
          *(_QWORD *)(a1 + 8) = v42;
          *(_QWORD *)(a1 + 16) = 1;
        }
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 34:
        sub_392BD70(a1, (_QWORD *)a2);
        return a1;
      case 35:
        v40 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 37;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v40;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 36:
        v39 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 26;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v39;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 37:
        v29 = *(_BYTE *)(*(_QWORD *)(a2 + 136) + 402LL);
        if ( !v29 )
          goto LABEL_100;
        v30 = *(_QWORD *)(a2 + 144);
        v31 = 0;
        v32 = 0;
        if ( !v30 )
          goto LABEL_118;
        v33 = strlen(*(const char **)(a2 + 144));
        v34 = v33;
        v32 = v33;
        if ( v33 <= 5 )
        {
          v31 = v33 > 2;
        }
        else
        {
          if ( *(_DWORD *)v30 == 1819042147 && *(_WORD *)(v30 + 4) == 13873 )
          {
            v37 = 7;
            v36 = 46;
            goto LABEL_58;
          }
          v31 = v33 > 2;
          if ( v33 > 6 )
          {
            if ( *(_DWORD *)v30 == 1819042147 && *(_WORD *)(v30 + 4) == 26719 )
            {
              v36 = 47;
              if ( *(_BYTE *)(v30 + 6) == 105 )
                goto LABEL_54;
            }
            v73 = v32;
            v35 = memcmp((const void *)v30, "call_lo", 7u);
            v31 = v34 > 2;
            v32 = v73;
            if ( !v35 )
            {
              v36 = 48;
LABEL_54:
              v37 = 8;
LABEL_58:
              v38 = *(_QWORD *)(a2 + 104);
              *(_QWORD *)(a2 + 144) = v37 - 1 + v30;
              *(_DWORD *)a1 = v36;
              *(_QWORD *)(a1 + 8) = v38;
              *(_QWORD *)(a1 + 16) = v37;
              *(_DWORD *)(a1 + 32) = 64;
              *(_QWORD *)(a1 + 24) = 0;
              return a1;
            }
            if ( v34 > 8 )
            {
              if ( *(_QWORD *)v30 == 0x685F6C6572707464LL )
              {
                v36 = 49;
                if ( *(_BYTE *)(v30 + 8) == 105 )
                  goto LABEL_115;
              }
              v68 = memcmp((const void *)v30, "dtprel_lo", 9u);
              v69 = v34 > 7;
              v32 = v73;
              if ( !v68 )
              {
                v36 = 50;
LABEL_115:
                v37 = 10;
                goto LABEL_58;
              }
              goto LABEL_145;
            }
            if ( v34 > 7 )
            {
              v69 = v29;
LABEL_145:
              if ( *(_QWORD *)v30 == 0x707369645F746F67LL )
              {
                v37 = 9;
                v36 = 52;
                goto LABEL_58;
              }
              goto LABEL_151;
            }
          }
        }
        v69 = 0;
        if ( v34 <= 5 )
        {
LABEL_118:
          v70 = 0;
          goto LABEL_119;
        }
LABEL_151:
        if ( *(_DWORD *)v30 != 1601466215 || (v71 = 0, *(_WORD *)(v30 + 4) != 26984) )
          v71 = 1;
        if ( !v71 )
        {
          v36 = 53;
LABEL_155:
          v37 = 7;
          goto LABEL_58;
        }
        v74 = v32;
        v75 = v69;
        v72 = memcmp((const void *)v30, "got_lo", 6u);
        v70 = v75;
        v31 = v34 > 2;
        v32 = v74;
        if ( !v72 )
        {
          v36 = 54;
          goto LABEL_155;
        }
        if ( v75 )
        {
          switch ( *(_QWORD *)v30 )
          {
            case 0x7473666F5F746F67LL:
              v37 = 9;
              v36 = 55;
              goto LABEL_58;
            case 0x656761705F746F67LL:
              v37 = 9;
              v36 = 56;
              goto LABEL_58;
            case 0x6C65727074746F67LL:
              v37 = 9;
              v36 = 57;
              goto LABEL_58;
          }
          v31 = v34 > 2;
        }
LABEL_119:
        if ( v31 && *(_WORD *)v30 == 28519 && *(_BYTE *)(v30 + 2) == 116 )
        {
          v37 = 4;
          v36 = 51;
          goto LABEL_58;
        }
        if ( v32 <= 5 )
          goto LABEL_129;
        if ( *(_DWORD *)v30 == 1918857319 && *(_WORD *)(v30 + 4) == 27749 )
        {
          v37 = 7;
          v36 = 58;
          goto LABEL_58;
        }
        if ( *(_DWORD *)v30 == 1751607656 && *(_WORD *)(v30 + 4) == 29285 )
        {
          v37 = 7;
          v36 = 60;
          goto LABEL_58;
        }
        if ( v32 <= 6 )
        {
LABEL_129:
          if ( v32 <= 1 )
            goto LABEL_130;
        }
        else if ( *(_DWORD *)v30 == 1751607656 && *(_WORD *)(v30 + 4) == 29541 && *(_BYTE *)(v30 + 6) == 116 )
        {
          v37 = 8;
          v36 = 61;
          goto LABEL_58;
        }
        if ( *(_WORD *)v30 == 26984 )
        {
          v36 = 59;
LABEL_128:
          v37 = 3;
          goto LABEL_58;
        }
        if ( *(_WORD *)v30 == 28524 )
        {
          v36 = 62;
          goto LABEL_128;
        }
LABEL_130:
        if ( v31 && *(_WORD *)v30 == 25966 && *(_BYTE *)(v30 + 2) == 103 )
        {
          v37 = 4;
          v36 = 63;
          goto LABEL_58;
        }
        if ( v70 )
        {
          if ( *(_QWORD *)v30 == 0x69685F6C65726370LL )
          {
            v36 = 64;
LABEL_135:
            v37 = 9;
            goto LABEL_58;
          }
          if ( *(_QWORD *)v30 == 0x6F6C5F6C65726370LL )
          {
            v36 = 65;
            goto LABEL_135;
          }
        }
        if ( v32 > 4 )
        {
          if ( *(_DWORD *)v30 == 1735617652 && *(_BYTE *)(v30 + 4) == 100 )
          {
            v37 = 6;
            v36 = 66;
            goto LABEL_58;
          }
          if ( v32 > 5 && *(_DWORD *)v30 == 1819503732 && *(_WORD *)(v30 + 4) == 28004 )
          {
            v37 = 7;
            v36 = 67;
            goto LABEL_58;
          }
        }
        if ( v70 )
        {
          if ( *(_QWORD *)v30 == 0x69685F6C65727074LL )
          {
            v37 = 9;
            v36 = 68;
            goto LABEL_58;
          }
          if ( *(_QWORD *)v30 == 0x6F6C5F6C65727074LL )
          {
            v37 = 9;
            v36 = 69;
            goto LABEL_58;
          }
        }
LABEL_100:
        v66 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 36;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v66;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 38:
        v27 = *(_BYTE **)(a2 + 144);
        v28 = *(_QWORD *)(a2 + 104);
        if ( *v27 == 38 )
        {
          *(_QWORD *)(a2 + 144) = v27 + 1;
          *(_DWORD *)a1 = 33;
          *(_QWORD *)(a1 + 8) = v28;
          *(_QWORD *)(a1 + 16) = 2;
        }
        else
        {
          *(_DWORD *)a1 = 32;
          *(_QWORD *)(a1 + 8) = v28;
          *(_QWORD *)(a1 + 16) = 1;
        }
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 39:
        sub_392BB70(a1, (_QWORD *)a2);
        return a1;
      case 40:
        v26 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 17;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v26;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 41:
        v25 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 18;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v25;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 42:
        v24 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 23;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v24;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 43:
        v65 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 12;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v65;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 44:
        v63 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 25;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v63;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 45:
        v57 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 13;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v57;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 47:
        *(_BYTE *)(a2 + 169) = v6;
        sub_392AEB0(a1, a2);
        return a1;
      case 48:
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 55:
      case 56:
      case 57:
        sub_392B090(a1, a2);
        return a1;
      case 58:
        v64 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 10;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v64;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 60:
        v60 = *(char **)(a2 + 144);
        v61 = *(_QWORD *)(a2 + 104);
        v62 = *v60;
        if ( *v60 == 61 )
        {
          *(_QWORD *)(a2 + 144) = v60 + 1;
          *(_DWORD *)a1 = 39;
          *(_QWORD *)(a1 + 8) = v61;
          *(_QWORD *)(a1 + 16) = 2;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
        }
        else if ( v62 == 62 )
        {
          *(_QWORD *)(a2 + 144) = v60 + 1;
          *(_DWORD *)a1 = 41;
          *(_QWORD *)(a1 + 8) = v61;
          *(_QWORD *)(a1 + 16) = 2;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
        }
        else
        {
          if ( v62 == 60 )
          {
            *(_QWORD *)(a2 + 144) = v60 + 1;
            *(_DWORD *)a1 = 40;
            *(_QWORD *)(a1 + 8) = v61;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 38;
            *(_QWORD *)(a1 + 8) = v61;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
        }
        return a1;
      case 61:
        v58 = *(_BYTE **)(a2 + 144);
        v59 = *(_QWORD *)(a2 + 104);
        if ( *v58 == 61 )
        {
          *(_QWORD *)(a2 + 144) = v58 + 1;
          *(_DWORD *)a1 = 28;
          *(_QWORD *)(a1 + 8) = v59;
          *(_QWORD *)(a1 + 16) = 2;
        }
        else
        {
          *(_DWORD *)a1 = 27;
          *(_QWORD *)(a1 + 8) = v59;
          *(_QWORD *)(a1 + 16) = 1;
        }
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 62:
        v55 = *(_BYTE **)(a2 + 144);
        v56 = *(_QWORD *)(a2 + 104);
        if ( *v55 == 61 )
        {
          *(_QWORD *)(a2 + 144) = v55 + 1;
          *(_DWORD *)a1 = 43;
          *(_QWORD *)(a1 + 8) = v56;
          *(_QWORD *)(a1 + 16) = 2;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
        }
        else
        {
          if ( *v55 == 62 )
          {
            *(_QWORD *)(a2 + 144) = v55 + 1;
            *(_DWORD *)a1 = 44;
            *(_QWORD *)(a1 + 8) = v56;
            *(_QWORD *)(a1 + 16) = 2;
          }
          else
          {
            *(_DWORD *)a1 = 42;
            *(_QWORD *)(a1 + 8) = v56;
            *(_QWORD *)(a1 + 16) = 1;
          }
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = 0;
        }
        return a1;
      case 64:
        v54 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 45;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v54;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 91:
        v53 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 19;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v53;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 92:
        v52 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 16;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v52;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 93:
        v51 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 20;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v51;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 94:
        v50 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 31;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v50;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 123:
        v49 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 21;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v49;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 124:
        v47 = *(_BYTE **)(a2 + 144);
        v48 = *(_QWORD *)(a2 + 104);
        if ( *v47 == 124 )
        {
          *(_QWORD *)(a2 + 144) = v47 + 1;
          *(_DWORD *)a1 = 30;
          *(_QWORD *)(a1 + 8) = v48;
          *(_QWORD *)(a1 + 16) = 2;
        }
        else
        {
          *(_DWORD *)a1 = 29;
          *(_QWORD *)(a1 + 8) = v48;
          *(_QWORD *)(a1 + 16) = 1;
        }
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 125:
        v23 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 22;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v23;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      case 126:
        v43 = *(_QWORD *)(a2 + 104);
        *(_DWORD *)a1 = 14;
        *(_QWORD *)(a1 + 16) = 1;
        *(_QWORD *)(a1 + 8) = v43;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      default:
        if ( isalpha(v4) || v4 == 95 || v4 == 46 )
        {
          sub_392ABD0(a1, a2);
        }
        else
        {
          v76 = 26;
          v77 = &v79;
          v14 = (unsigned __int64 *)sub_22409D0((__int64)&v77, &v76, 0);
          si128 = _mm_load_si128((const __m128i *)&xmmword_3F90190);
          v77 = v14;
          v79 = v76;
          qmemcpy(v14 + 2, "r in input", 10);
          *(__m128i *)v14 = si128;
          v78 = v76;
          *((_BYTE *)v77 + v76) = 0;
          sub_392A760(a1, (_QWORD *)a2, *(_QWORD *)(a2 + 104), (unsigned __int64 *)&v77);
          if ( v77 != &v79 )
            j_j___libc_free_0((unsigned __int64)v77);
        }
        return a1;
    }
  }
  if ( v6 )
  {
    v9 = *(_QWORD *)(a2 + 104);
    *(_WORD *)(a2 + 168) = 257;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  else
  {
    v46 = *(_QWORD *)(a2 + 104);
    *(_WORD *)(a2 + 168) = 257;
LABEL_71:
    *(_DWORD *)a1 = 9;
    *(_QWORD *)(a1 + 8) = v46;
    *(_QWORD *)(a1 + 16) = 1;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  return a1;
}
