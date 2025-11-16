// Function: sub_739430
// Address: 0x739430
//
__int64 __fastcall sub_739430(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, _UNKNOWN *__ptr32 *a5)
{
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rsi
  unsigned int v8; // r12d
  __int64 v9; // r13
  char v10; // cl
  char v11; // r10
  int v12; // edx
  __int64 v13; // rcx
  int v14; // r10d
  unsigned int v15; // r15d
  char v16; // al
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  bool v20; // dl
  __int64 v21; // r11
  char v22; // al
  int v23; // eax
  __int128 v24; // rdi
  unsigned __int8 v25; // al
  char v26; // dl
  unsigned __int8 v27; // al
  bool v28; // zf
  __int64 v29; // rdi
  __int64 v30; // rsi
  unsigned int v31; // r9d
  char v32; // al
  __int64 v33; // rcx
  __int64 v34; // rbx
  __int64 v35; // r13
  size_t v36; // rdx
  char v37; // al
  int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // eax
  char v44; // dl
  int v45; // eax
  char v46; // dl
  int v47; // eax
  int v48; // eax
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 *v51; // r15
  __int128 v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int128 v58; // rdi
  int v59; // eax
  __int128 v60; // rdi
  int v61; // eax
  char v62; // dl
  __int64 v63; // rcx
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r9
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // [rsp+8h] [rbp-58h]
  bool v73; // [rsp+8h] [rbp-58h]
  __int64 v74; // [rsp+8h] [rbp-58h]
  bool v75; // [rsp+8h] [rbp-58h]
  __int64 v76; // [rsp+10h] [rbp-50h]
  char v77; // [rsp+10h] [rbp-50h]
  __int64 v78; // [rsp+10h] [rbp-50h]
  int v79; // [rsp+18h] [rbp-48h]
  __int64 v80; // [rsp+18h] [rbp-48h]
  int v81; // [rsp+24h] [rbp-3Ch]
  int v82; // [rsp+28h] [rbp-38h]
  int v83; // [rsp+28h] [rbp-38h]
  __int64 savedregs; // [rsp+60h] [rbp+0h] BYREF

  while ( 2 )
  {
    v5 = a2;
    v6 = *(_QWORD *)(a1 + 128);
    v7 = *(_QWORD *)(a2 + 128);
    if ( v5 == a1 )
      return 1;
    v8 = a3;
    v9 = a1;
    v10 = a3;
    v11 = a3;
    v82 = a3 & 3;
    v12 = v82 != 0 ? 0x40 : 0;
    BYTE1(v12) = 1;
    v13 = v10 & 4;
    if ( !(_DWORD)v13 )
      v12 = v82 != 0 ? 0x40 : 0;
    v81 = v13;
    v14 = v11 & 1;
    v15 = v12;
LABEL_5:
    v16 = *(_BYTE *)(v9 + 173);
    if ( v16 != *(_BYTE *)(v5 + 173) )
      return 0;
    if ( v6 == 0 || v7 == 0 )
    {
      if ( v6 != v7 )
        return 0;
      goto LABEL_8;
    }
    if ( !v14 )
    {
      v80 = sub_8D2220(v6);
      v19 = sub_8D2220(v7);
      v20 = v6 == 0 || v7 == 0;
      v21 = v19;
      if ( *(_BYTE *)(v9 + 173) == 6
        && *(_BYTE *)(v9 + 176) == 1
        && *(_BYTE *)(v5 + 176) == 1
        && (*(_BYTE *)(v9 + 168) & 8) == 0
        && (*(_BYTE *)(v5 + 168) & 8) == 0 )
      {
        v72 = v19;
        v38 = sub_8D2E30(v80);
        v20 = v6 == 0 || v7 == 0;
        v21 = v72;
        if ( v38 )
        {
          v39 = sub_8D2E30(v72);
          v21 = v72;
          v20 = v6 == 0 || v7 == 0;
          if ( v39 )
          {
            v40 = sub_8D46C0(v80);
            v80 = sub_8D2220(v40);
            v41 = sub_8D46C0(v72);
            v42 = sub_8D2220(v41);
            v20 = v6 == 0 || v7 == 0;
            v21 = v42;
          }
        }
      }
      if ( dword_4F077BC )
      {
        v22 = *(_BYTE *)(v7 + 140);
        if ( *(_BYTE *)(v6 + 140) != v22 && v22 == 14 )
        {
          v75 = v20;
          v78 = v21;
          v48 = sub_8DBA70(v6, v21);
          v21 = v78;
          v20 = v75;
          a5 = jpt_7394DD;
          v14 = 0;
          if ( v48 )
            goto LABEL_33;
        }
      }
      if ( (v8 & 8) == 0 )
      {
LABEL_32:
        v23 = sub_8DED30(v80, v21, 4);
        v14 = 0;
        a5 = jpt_7394DD;
        if ( !v23 )
          return 0;
        goto LABEL_33;
      }
      v73 = v20;
      v76 = v21;
      v43 = sub_8D3D40(v21);
      v44 = v73;
      if ( !v43 || *(_BYTE *)(v76 + 140) != 14 || *(_BYTE *)(v76 + 160) || *(_DWORD *)(*(_QWORD *)(v76 + 168) + 28LL) )
      {
        v74 = v76;
        v77 = v44;
        v47 = sub_8D3D40(v80);
        v46 = v77;
        v21 = v74;
        a5 = jpt_7394DD;
        v14 = 0;
        if ( !v47 )
          goto LABEL_85;
        if ( *(_BYTE *)(v80 + 140) != 14 )
        {
LABEL_97:
          if ( !dword_4F077BC )
            goto LABEL_32;
          goto LABEL_86;
        }
        if ( *(_BYTE *)(v80 + 160) )
        {
LABEL_85:
          v13 = dword_4F077BC;
          if ( !dword_4F077BC )
            goto LABEL_32;
LABEL_86:
          if ( (_DWORD)qword_4F077B4 )
          {
            if ( qword_4F077A0 <= 0x784Fu )
              goto LABEL_32;
          }
          else if ( qword_4F077A8 <= 0xEA5Fu )
          {
            goto LABEL_32;
          }
          if ( !v46 )
            goto LABEL_32;
LABEL_33:
          v16 = *(_BYTE *)(v9 + 173);
          goto LABEL_8;
        }
      }
      else
      {
        v45 = sub_8D3D40(v80);
        v21 = v76;
        v14 = 0;
        a5 = jpt_7394DD;
        if ( !v45 || *(_BYTE *)(v80 + 140) != 14 || *(_BYTE *)(v80 + 160) )
          goto LABEL_33;
        v46 = 1;
      }
      if ( !*(_DWORD *)(*(_QWORD *)(v80 + 168) + 28LL) )
      {
        if ( v46 || !dword_4F077BC )
          goto LABEL_33;
        v46 = 1;
        goto LABEL_86;
      }
      if ( v46 )
        goto LABEL_33;
      goto LABEL_97;
    }
    if ( v6 != v7 )
    {
      if ( !dword_4F07588 || (v18 = *(_QWORD *)(v6 + 32), *(_QWORD *)(v7 + 32) != v18) || !v18 )
      {
        v79 = v14;
        if ( v16 != 2 || !(unsigned int)sub_8D97D0(v6, v7, v15, v13, a5) )
          return 0;
        v16 = *(_BYTE *)(v9 + 173);
        v14 = v79;
        a5 = jpt_7394DD;
      }
    }
LABEL_8:
    switch ( v16 )
    {
      case 0:
      case 14:
        return 1;
      case 1:
        LOBYTE(result) = (unsigned int)sub_621060(v9, v5) == 0;
        if ( (v8 & 1) == 0 || !(_BYTE)result )
          return (unsigned __int8)result;
        return ((*(_BYTE *)(v5 + 169) ^ *(_BYTE *)(v9 + 169)) & 0x19) == 0;
      case 2:
        v36 = *(_QWORD *)(v9 + 176);
        if ( v36 != *(_QWORD *)(v5 + 176) )
          return 0;
        return memcmp(*(const void **)(v9 + 184), *(const void **)(v5 + 184), v36) == 0;
      case 3:
      case 5:
        while ( *(_BYTE *)(v6 + 140) == 12 )
          v6 = *(_QWORD *)(v6 + 160);
        if ( !(unsigned int)sub_8D2A90(v6) )
          return 0;
        return sub_70C900(*(unsigned __int8 *)(v6 + 160), (const void *)(v9 + 176), (const void *)(v5 + 176));
      case 4:
        while ( *(_BYTE *)(v6 + 140) == 12 )
          v6 = *(_QWORD *)(v6 + 160);
        if ( !(unsigned int)sub_8D2A90(v6) )
          return 0;
        result = sub_70C900(*(unsigned __int8 *)(v6 + 160), *(const void **)(v9 + 176), *(const void **)(v5 + 176));
        if ( (_DWORD)result )
          return sub_70C900(
                   *(unsigned __int8 *)(v6 + 160),
                   (const void *)(*(_QWORD *)(v9 + 176) + 16LL),
                   (const void *)(*(_QWORD *)(v5 + 176) + 16LL));
        return result;
      case 6:
        v37 = *(_BYTE *)(v9 + 176);
        if ( v37 == *(_BYTE *)(v5 + 176) )
        {
          v13 = *(_QWORD *)(v5 + 192);
          if ( *(_QWORD *)(v9 + 192) == v13 )
          {
            switch ( v37 )
            {
              case 0:
                v29 = *(_QWORD *)(v9 + 184);
                v30 = *(_QWORD *)(v5 + 184);
                result = 1;
                if ( v29 == v30 )
                  return result;
                goto LABEL_46;
              case 1:
                v29 = *(_QWORD *)(v9 + 184);
                v30 = *(_QWORD *)(v5 + 184);
                result = 1;
                if ( v29 == v30 )
                  return result;
                if ( v29 != 0 && v30 != 0 && *qword_4D03FD0 )
                  return (unsigned int)sub_8C7EB0(v29, v30) != 0;
                return 0;
              case 2:
                v9 = *(_QWORD *)(v9 + 184);
                v5 = *(_QWORD *)(v5 + 184);
                if ( (v8 & 0x10) == 0 )
                  return v5 == v9;
                goto LABEL_41;
              case 3:
              case 6:
                goto LABEL_36;
              case 5:
                v56 = *(_QWORD *)(v9 + 184);
                v57 = *(_QWORD *)(v5 + 184);
                result = 1;
                if ( v56 != v57 )
                  return (unsigned int)sub_8D97D0(v56, v57, v15, v13, a5) != 0;
                return result;
              default:
                goto LABEL_129;
            }
          }
        }
        return 0;
      case 7:
        v27 = *(_BYTE *)(v9 + 192);
        if ( ((v27 ^ *(_BYTE *)(v5 + 192)) & 2) != 0 )
          return 0;
        v28 = (v27 & 2) == 0;
        v29 = *(_QWORD *)(v9 + 200);
        v30 = *(_QWORD *)(v5 + 200);
        result = 1;
        if ( v28 )
        {
          if ( v29 == v30 )
            return result;
          if ( v29 != 0 && v30 != 0 && *qword_4D03FD0 )
            return (unsigned int)sub_8C7EB0(v29, v30) != 0;
        }
        else
        {
          if ( v29 == v30 )
            return result;
LABEL_46:
          if ( v29 != 0 && v30 != 0 && *qword_4D03FD0 )
            return (unsigned int)sub_8C7EB0(v29, v30) != 0;
        }
        return 0;
      case 8:
        result = sub_739430(*(_QWORD *)(v9 + 176), *(_QWORD *)(v5 + 176), v8, v13);
        if ( (_DWORD)result )
          return (unsigned int)sub_739430(*(_QWORD *)(v9 + 184), *(_QWORD *)(v5 + 184), v8, v33) != 0;
        return result;
      case 9:
        *((_QWORD *)&v24 + 1) = *(_QWORD *)(v5 + 176);
        *(_QWORD *)&v24 = *(_QWORD *)(v9 + 176);
        if ( v24 == 0 )
          return 1;
        if ( !(_QWORD)v24 || !*((_QWORD *)&v24 + 1) )
          return 0;
        a5 = (_UNKNOWN *__ptr32 *)v8;
        result = 0;
        v62 = *(_BYTE *)(v24 + 48);
        if ( v62 != *(_BYTE *)(*((_QWORD *)&v24 + 1) + 48LL)
          || ((*(_BYTE *)(*((_QWORD *)&v24 + 1) + 49LL) ^ *(_BYTE *)(v24 + 49)) & 0x11) != 0 )
        {
          return result;
        }
        v63 = *(_QWORD *)(v24 + 8);
        v64 = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 8LL);
        if ( v63 == v64 )
          goto LABEL_185;
        if ( !v63 || !v64 )
          return 0;
        result = dword_4F07588;
        if ( !dword_4F07588 )
          return result;
        v65 = *(_QWORD *)(v63 + 32);
        if ( *(_QWORD *)(v64 + 32) != v65 || !v65 )
          return 0;
LABEL_185:
        v66 = *(_QWORD *)(v24 + 16);
        v67 = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 16LL);
        if ( v66 == v67 )
          goto LABEL_191;
        if ( !v66 || !v67 )
          return 0;
        result = dword_4F07588;
        if ( dword_4F07588 )
        {
          v68 = *(_QWORD *)(v66 + 32);
          if ( *(_QWORD *)(v67 + 32) == v68 && v68 )
          {
LABEL_191:
            savedregs = (__int64)&savedregs;
            switch ( v62 )
            {
              case 0:
              case 1:
                return 1;
              case 2:
              case 6:
                a2 = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 56LL);
                a1 = *(_QWORD *)(v24 + 56);
                a3 = v8;
                continue;
              case 3:
              case 4:
                *((_QWORD *)&v24 + 1) = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 56LL);
                *(_QWORD *)&v24 = *(_QWORD *)(v24 + 56);
                return sub_7386E0(v24, v8);
              case 5:
                v69 = *(_QWORD *)(v24 + 56);
                v70 = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 56LL);
                if ( v69 == v70 )
                  goto LABEL_203;
                if ( !v70 || !v69 )
                  goto LABEL_202;
                result = dword_4F07588;
                if ( dword_4F07588 )
                {
                  v71 = *(_QWORD *)(v69 + 32);
                  if ( *(_QWORD *)(v70 + 32) == v71 && v71 )
                  {
LABEL_203:
                    result = 0;
                    if ( ((*(_BYTE *)(*((_QWORD *)&v24 + 1) + 72LL) ^ *(_BYTE *)(v24 + 72)) & 4) == 0 )
                    {
                      *((_QWORD *)&v24 + 1) = *(_QWORD *)(*((_QWORD *)&v24 + 1) + 64LL);
                      *(_QWORD *)&v24 = *(_QWORD *)(v24 + 64);
                      result = (unsigned int)sub_739370(v24, v8) != 0;
                    }
                  }
                  else
                  {
LABEL_202:
                    result = 0;
                  }
                }
                break;
              default:
                sub_721090();
            }
          }
          else
          {
            return 0;
          }
        }
        return result;
      case 10:
        v34 = *(_QWORD *)(v9 + 176);
        v35 = *(_QWORD *)(v5 + 176);
        if ( !v34 )
          return (v34 | v35) == 0;
        do
        {
          if ( !v35 )
            break;
          if ( !(unsigned int)sub_739430(v34, v35, v8, v13) )
            return 0;
          v34 = *(_QWORD *)(v34 + 120);
          v35 = *(_QWORD *)(v35 + 120);
        }
        while ( v34 );
        return (v34 | v35) == 0;
      case 11:
        result = sub_739430(*(_QWORD *)(v9 + 176), *(_QWORD *)(v5 + 176), v8, v13);
        if ( (_DWORD)result )
        {
          result = 0;
          if ( *(_QWORD *)(v9 + 184) == *(_QWORD *)(v5 + 184) )
            return *(_BYTE *)(v9 + 192) == *(_BYTE *)(v5 + 192);
        }
        return result;
      case 12:
        v31 = v15;
        v32 = *(_BYTE *)(v9 + 176);
        if ( v32 == *(_BYTE *)(v5 + 176) )
        {
          switch ( v32 )
          {
            case 0:
              if ( (v8 & 2) != 0 )
                return 0;
              result = 1;
              if ( (v8 & 8) != 0 )
                return result;
              if ( *(_DWORD *)(v9 + 184) != *(_DWORD *)(v5 + 184) )
                return 0;
              v61 = *(_DWORD *)(v5 + 188);
              if ( *(_DWORD *)(v9 + 188) != v61 && *(_DWORD *)(v9 + 188) != 0 )
              {
                if ( v61 )
                  return 0;
              }
              return 1;
            case 1:
              *((_QWORD *)&v60 + 1) = sub_72E9A0(v5);
              *(_QWORD *)&v60 = sub_72E9A0(v9);
              result = sub_7386E0(v60, v8);
              goto LABEL_108;
            case 2:
              if ( *(_QWORD *)(v9 + 8) != *(_QWORD *)(v5 + 8) )
                return 0;
              result = sub_728940(v9, v5, v8);
              goto LABEL_108;
            case 3:
              if ( *(_QWORD *)(v9 + 8) != *(_QWORD *)(v5 + 8) )
                return 0;
              v83 = v14;
              if ( !sub_728940(v9, v5, v8) )
                return 0;
              v31 = v15;
              if ( v83 )
              {
                if ( *(_QWORD *)(v9 + 192) != *(_QWORD *)(v5 + 192) )
                  return 0;
              }
              if ( *(_BYTE *)(v9 + 200) != *(_BYTE *)(v5 + 200)
                || ((*(_BYTE *)(v5 + 177) ^ *(_BYTE *)(v9 + 177)) & 1) != 0 )
              {
                return 0;
              }
              *(_QWORD *)&v58 = *(_QWORD *)(v9 + 184);
              *((_QWORD *)&v58 + 1) = *(_QWORD *)(v5 + 184);
              if ( (_QWORD)v58 && *((_QWORD *)&v58 + 1) )
              {
LABEL_139:
                if ( (_QWORD)v58 != *((_QWORD *)&v58 + 1)
                  && !(unsigned int)sub_8D97D0(v58, *((_QWORD *)&v58 + 1), v31, v13, a5) )
                {
                  return 0;
                }
              }
              else if ( v58 != 0 )
              {
                return 0;
              }
LABEL_141:
              result = 1;
LABEL_109:
              if ( (v8 & 2) == 0 )
                return result;
              if ( ((*(_BYTE *)(v9 + 177) ^ *(_BYTE *)(v5 + 177)) & 0x10) == 0 )
              {
                if ( (*(_BYTE *)(v9 + 177) & 0x10) == 0 )
                  return result;
                v53 = *(_QWORD *)(v9 + 48);
                v54 = *(_QWORD *)(v5 + 48);
                if ( v53 == v54 )
                  return result;
                if ( v53 )
                {
                  if ( v54 )
                  {
                    if ( dword_4F07588 )
                    {
                      v55 = *(_QWORD *)(v53 + 32);
                      if ( *(_QWORD *)(v54 + 32) == v55 )
                      {
                        if ( v55 )
                          return result;
                      }
                    }
                  }
                }
              }
              return 0;
            case 4:
            case 12:
              result = sub_739430(*(_QWORD *)(v9 + 184), *(_QWORD *)(v5 + 184), v8, v13);
              goto LABEL_108;
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
            case 10:
              v49 = *(_QWORD *)(v9 + 184);
              v50 = *(_QWORD *)(v5 + 184);
              if ( v49 != v50 && !(unsigned int)sub_8D97D0(v49, v50, v15, v13, a5) )
                return 0;
              v51 = sub_72F1F0(v9);
              *((_QWORD *)&v52 + 1) = sub_72F1F0(v5);
              if ( !(*((_QWORD *)&v52 + 1) | (unsigned __int64)v51) )
                goto LABEL_141;
              if ( v51 && *((_QWORD *)&v52 + 1) )
              {
                *(_QWORD *)&v52 = v51;
                result = sub_7386E0(v52, v8);
LABEL_108:
                if ( (_DWORD)result )
                  goto LABEL_109;
              }
              return 0;
            case 11:
              if ( !(unsigned int)sub_739430(*(_QWORD *)(v9 + 184), *(_QWORD *)(v5 + 184), v8, v13) )
                return 0;
              v59 = 16 * (v82 != 0);
              if ( v81 )
                v59 = (16 * (v82 != 0)) | 0x40;
              if ( !(unsigned int)sub_89AB40(*(_QWORD *)(v9 + 192), *(_QWORD *)(v5 + 192), v59 | 2u) )
                return 0;
              goto LABEL_141;
            case 13:
              *(_QWORD *)&v58 = *(_QWORD *)(v9 + 184);
              *((_QWORD *)&v58 + 1) = *(_QWORD *)(v5 + 184);
              goto LABEL_139;
            default:
              goto LABEL_129;
          }
        }
        return 0;
      case 13:
        v25 = *(_BYTE *)(v9 + 176);
        if ( ((v25 ^ *(_BYTE *)(v5 + 176)) & 3) != 0 )
          return 0;
        v9 = *(_QWORD *)(v9 + 184);
        v5 = *(_QWORD *)(v5 + 184);
        v26 = v25 & 1;
        if ( (v25 & 2) != 0 )
        {
          if ( v26 )
            return strcmp((const char *)v9, (const char *)v5) == 0;
LABEL_41:
          v6 = *(_QWORD *)(v9 + 128);
          v7 = *(_QWORD *)(v5 + 128);
          if ( v9 == v5 )
            return 1;
          goto LABEL_5;
        }
        if ( !v26 )
          return v5 == v9;
        if ( v9 == v5 )
        {
          LOBYTE(result) = 1;
        }
        else
        {
          LOBYTE(result) = v9 != 0 && v5 != 0;
          if ( (_BYTE)result )
          {
            LOBYTE(result) = 0;
            if ( dword_4F07588 )
              LOBYTE(result) = *(_QWORD *)(v9 + 32) != 0 && *(_QWORD *)(v5 + 32) == *(_QWORD *)(v9 + 32);
          }
        }
        return (unsigned __int8)result;
      case 15:
LABEL_36:
        LOBYTE(result) = *(_QWORD *)(v9 + 184) == *(_QWORD *)(v5 + 184);
        return (unsigned __int8)result;
      default:
LABEL_129:
        sub_721090();
    }
  }
}
