// Function: sub_1131D50
// Address: 0x1131d50
//
char __fastcall sub_1131D50(__int64 a1, char a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r13
  unsigned int v5; // r12d
  unsigned __int8 v6; // bl
  unsigned __int8 v7; // al
  unsigned int v8; // r14d
  __int64 v9; // r15
  int v11; // edx
  char result; // al
  __int64 v13; // r14
  _BYTE *v14; // rax
  char v15; // dl
  _BYTE *v16; // r15
  unsigned int v17; // r14d
  __int64 *v18; // rdi
  int v19; // r15d
  unsigned int v20; // eax
  unsigned int *v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // r12d
  __int64 v24; // rax
  unsigned int v25; // r14d
  __int64 v26; // r15
  bool v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  _BYTE *v30; // rax
  _BYTE *v31; // r15
  unsigned int v32; // r14d
  __int64 v33; // rdi
  int v34; // eax
  bool v35; // al
  unsigned __int64 v37; // rdx
  int v38; // r14d
  unsigned int v39; // r15d
  _BYTE *v40; // rax
  _BYTE *v41; // rsi
  char v42; // al
  unsigned int v43; // r10d
  int v44; // eax
  bool v45; // r14
  unsigned int i; // r15d
  _BYTE *v47; // rax
  _BYTE *v48; // rdx
  char v49; // al
  __int64 v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rax
  __int64 *v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rsi
  char v59; // al
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned int v64; // eax
  unsigned __int64 v66; // rax
  int v68; // r15d
  int v70; // esi
  unsigned __int64 v71; // rcx
  int v73; // r14d
  __int64 v74; // rax
  char v75; // al
  __int64 v76; // rsi
  unsigned __int8 *v77; // rdx
  __int64 v78; // rax
  _BYTE *v79; // rsi
  _BYTE *v80; // rax
  __int64 v81; // rdx
  unsigned int v82; // [rsp+Ch] [rbp-84h]
  _BYTE *v83; // [rsp+10h] [rbp-80h]
  int v84; // [rsp+10h] [rbp-80h]
  char v85; // [rsp+18h] [rbp-78h]
  __int64 v86; // [rsp+18h] [rbp-78h]
  int v87; // [rsp+20h] [rbp-70h]
  __int64 *v88; // [rsp+20h] [rbp-70h]
  _BYTE *v89; // [rsp+20h] [rbp-70h]
  int v90; // [rsp+20h] [rbp-70h]
  char v91; // [rsp+20h] [rbp-70h]
  int v92; // [rsp+20h] [rbp-70h]
  int v93; // [rsp+20h] [rbp-70h]
  __int64 *v94; // [rsp+28h] [rbp-68h]
  __int64 v95; // [rsp+38h] [rbp-58h] BYREF
  __int64 *v96; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v97; // [rsp+48h] [rbp-48h] BYREF
  _QWORD *v98[8]; // [rsp+50h] [rbp-40h] BYREF

  while ( 2 )
  {
    v4 = a1;
    v5 = a4;
    v6 = a2;
    v94 = (__int64 *)a3;
    v7 = *(_BYTE *)a1;
    if ( !a2 )
    {
      if ( v7 == 17 )
      {
        v25 = *(_DWORD *)(a1 + 32);
        v26 = a1 + 24;
        if ( v25 <= 0x40 )
          v27 = *(_QWORD *)(a1 + 24) == 0;
        else
          v27 = v25 == (unsigned int)sub_C444A0(a1 + 24);
        if ( v27 )
          return 1;
        if ( v25 <= 0x40 )
        {
          v28 = *(_QWORD *)(a1 + 24);
          if ( !v28 )
            goto LABEL_18;
LABEL_44:
          if ( (v28 & (v28 + 1)) != 0 )
            goto LABEL_18;
          return 1;
        }
        v90 = sub_C445E0(v26);
        if ( v90 && v25 == (unsigned int)sub_C444A0(v26) + v90 )
          return 1;
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 8);
        v29 = (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17;
        if ( (unsigned int)v29 > 1 || v7 > 0x15u )
          goto LABEL_19;
        v30 = sub_AD7630(a1, 0, v29);
        v31 = v30;
        if ( v30 && *v30 == 17 )
        {
          v32 = *((_DWORD *)v30 + 8);
          v33 = (__int64)(v30 + 24);
          if ( v32 <= 0x40 )
          {
            v35 = *((_QWORD *)v30 + 3) == 0;
          }
          else
          {
            v89 = v30 + 24;
            v34 = sub_C444A0(v33);
            v33 = (__int64)v89;
            v35 = v32 == v34;
          }
          if ( v35 )
            return 1;
          if ( v32 <= 0x40 )
          {
            v28 = *((_QWORD *)v31 + 3);
            if ( !v28 )
              goto LABEL_18;
            goto LABEL_44;
          }
          v68 = sub_C445E0(v33);
          if ( v68 && v32 == (unsigned int)sub_C444A0(v33) + v68 )
            return 1;
        }
        else if ( *(_BYTE *)(v13 + 8) == 17 )
        {
          v92 = *(_DWORD *)(v13 + 32);
          if ( v92 )
          {
            v45 = a2;
            for ( i = 0; i != v92; ++i )
            {
              v47 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, i);
              v48 = v47;
              if ( !v47 )
                goto LABEL_18;
              v49 = *v47;
              if ( v49 != 13 )
              {
                if ( v49 != 17 )
                  goto LABEL_18;
                v83 = v48;
                v86 = (__int64)(v48 + 24);
                v45 = sub_9867B0((__int64)(v48 + 24));
                if ( !v45 )
                {
                  if ( *((_DWORD *)v83 + 8) > 0x40u )
                  {
                    v84 = *((_DWORD *)v83 + 8);
                    v73 = sub_C445E0(v86);
                    if ( !v73 || v84 != (unsigned int)sub_C444A0(v86) + v73 )
                      goto LABEL_18;
                  }
                  else
                  {
                    v50 = *((_QWORD *)v83 + 3);
                    if ( !v50 || (v50 & (v50 + 1)) != 0 )
                      goto LABEL_18;
                  }
                  v45 = 1;
                }
              }
            }
            if ( v45 )
              return 1;
          }
        }
      }
LABEL_18:
      v13 = *(_QWORD *)(v4 + 8);
      goto LABEL_19;
    }
    if ( v7 == 17 )
    {
      v8 = *(_DWORD *)(a1 + 32);
      v9 = a1 + 24;
      if ( v8 <= 0x40 )
      {
        if ( !*(_QWORD *)(a1 + 24) )
          return 1;
        _RAX = *(_QWORD *)(a1 + 24);
        if ( (_RAX & (1LL << ((unsigned __int8)v8 - 1))) == 0 )
          goto LABEL_18;
        if ( v8 )
        {
          v11 = 64;
          if ( _RAX << (64 - (unsigned __int8)v8) != -1 )
          {
            _BitScanReverse64(&v37, ~(_RAX << (64 - (unsigned __int8)v8)));
            v11 = v37 ^ 0x3F;
          }
        }
        else
        {
          v11 = 0;
        }
        __asm { tzcnt   rax, rax }
        if ( (unsigned int)_RAX > v8 )
          LODWORD(_RAX) = *(_DWORD *)(a1 + 32);
      }
      else
      {
        if ( v8 == (unsigned int)sub_C444A0(a1 + 24) )
          return 1;
        if ( (*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL * ((v8 - 1) >> 6)) & (1LL << ((unsigned __int8)v8 - 1))) == 0 )
          goto LABEL_18;
        v87 = sub_C44500(v9);
        LODWORD(_RAX) = sub_C44590(v9);
        v11 = v87;
      }
      if ( v8 != v11 + (_DWORD)_RAX )
        goto LABEL_18;
      return 1;
    }
    v13 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 && v7 <= 0x15u )
    {
      v14 = sub_AD7630(a1, 0, a3);
      v15 = a2;
      v16 = v14;
      if ( v14 && *v14 == 17 )
      {
        v17 = *((_DWORD *)v14 + 8);
        v18 = (__int64 *)(v14 + 24);
        if ( v17 <= 0x40 )
        {
          if ( !*((_QWORD *)v14 + 3) )
            return 1;
          if ( !sub_986C60(v18, v17 - 1) )
            goto LABEL_18;
          _RDX = *((_QWORD *)v16 + 3);
          if ( v17 )
          {
            v19 = 64;
            if ( _RDX << (64 - (unsigned __int8)v17) != -1 )
            {
              _BitScanReverse64(&v66, ~(_RDX << (64 - (unsigned __int8)v17)));
              v19 = v66 ^ 0x3F;
            }
          }
          else
          {
            v19 = 0;
          }
          v20 = 64;
          __asm { tzcnt   rcx, rdx }
          if ( _RDX )
            v20 = _RCX;
          if ( v17 <= v20 )
            v20 = v17;
        }
        else
        {
          v88 = (__int64 *)(v14 + 24);
          if ( v17 == (unsigned int)sub_C444A0((__int64)v18) )
            return 1;
          if ( !sub_986C60(v88, v17 - 1) )
            goto LABEL_18;
          v19 = sub_C44500((__int64)v88);
          v20 = sub_C44590((__int64)v88);
        }
        if ( v17 != v19 + v20 )
          goto LABEL_18;
        return 1;
      }
      if ( *(_BYTE *)(v13 + 8) == 17 )
      {
        v38 = *(_DWORD *)(v13 + 32);
        if ( v38 )
        {
          v85 = 0;
          v39 = 0;
          while ( 1 )
          {
            v91 = v15;
            v40 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a1, v39);
            v15 = v91;
            v41 = v40;
            if ( !v40 )
              goto LABEL_18;
            v42 = *v40;
            if ( v42 == 13 )
              goto LABEL_67;
            if ( v42 != 17 )
              goto LABEL_18;
            v43 = *((_DWORD *)v41 + 8);
            if ( v43 <= 0x40 )
              break;
            v85 = v91;
            v82 = *((_DWORD *)v41 + 8);
            v44 = sub_C444A0((__int64)(v41 + 24));
            v15 = v91;
            if ( v82 != v44 )
            {
              if ( (*(_QWORD *)(*((_QWORD *)v41 + 3) + 8LL * ((v82 - 1) >> 6)) & (1LL << ((unsigned __int8)v82 - 1))) == 0 )
                goto LABEL_18;
              v93 = sub_C44500((__int64)(v41 + 24));
              LODWORD(_RAX) = sub_C44590((__int64)(v41 + 24));
              v15 = v85;
              v43 = v82;
              v70 = v93;
LABEL_133:
              if ( v43 != v70 + (_DWORD)_RAX )
                goto LABEL_18;
              goto LABEL_134;
            }
LABEL_67:
            if ( v38 == ++v39 )
            {
              if ( !v85 )
                goto LABEL_18;
              return 1;
            }
          }
          _RAX = *((_QWORD *)v41 + 3);
          if ( _RAX )
          {
            if ( !_bittest64(&_RAX, v43 - 1) )
              goto LABEL_18;
            if ( v43 )
            {
              v70 = 64;
              if ( _RAX << (64 - (unsigned __int8)v43) != -1 )
              {
                _BitScanReverse64(&v71, ~(_RAX << (64 - (unsigned __int8)v43)));
                v70 = v71 ^ 0x3F;
              }
            }
            else
            {
              v70 = 0;
            }
            __asm { tzcnt   rax, rax }
            if ( (unsigned int)_RAX > v43 )
              LODWORD(_RAX) = v43;
            goto LABEL_133;
          }
LABEL_134:
          v85 = v15;
          goto LABEL_67;
        }
      }
      goto LABEL_18;
    }
LABEL_19:
    if ( (unsigned int)sub_BCB060(v13) == 1 )
      return 1;
    v21 = (unsigned int *)sub_C94E20((__int64)qword_4F862D0);
    if ( v21 )
      v22 = *v21;
    else
      v22 = qword_4F862D0[2];
    if ( v5 >= v22 || *(_BYTE *)v4 <= 0x1Cu )
      return 0;
    v23 = v5 + 1;
    switch ( *(_BYTE *)v4 )
    {
      case '*':
        if ( v6 )
          return 0;
        v96 = 0;
        v56 = sub_986520(v4);
        if ( !(unsigned __int8)sub_995B10(&v96, *(_QWORD *)(v56 + 32)) )
          return 0;
        v57 = (__int64 *)sub_986520(v4);
        return sub_9B64A0(*v57, *v94, 1u, v23, v94[4], v94[5], v94[3], 1);
      case ',':
        if ( !v6 )
          return 0;
        v54 = (__int64 *)sub_986520(v4);
        if ( !(unsigned __int8)sub_1112510(*v54) )
          return 0;
        v55 = sub_986520(v4);
        return sub_9B64A0(*(_QWORD *)(v55 + 32), *v94, 1u, v23, v94[4], v94[5], v94[3], 1);
      case '6':
        if ( !v6 )
          return 0;
        v53 = (__int64 *)sub_986520(v4);
        a3 = (__int64)v94;
        a4 = v23;
        a2 = 1;
        a1 = *v53;
        continue;
      case '7':
      case 'D':
        if ( v6 )
          return 0;
        v52 = (__int64 *)sub_986520(v4);
        a3 = (__int64)v94;
        a4 = v23;
        a2 = 0;
        a1 = *v52;
        continue;
      case '8':
      case 'E':
        goto LABEL_84;
      case '9':
      case ':':
        v24 = sub_986520(v4);
        if ( !(unsigned __int8)sub_1131D50(*(_QWORD *)(v24 + 32), v6, v94, v23) )
          return 0;
LABEL_84:
        v51 = (__int64 *)sub_986520(v4);
        a3 = (__int64)v94;
        a4 = v23;
        a2 = v6;
        a1 = *v51;
        continue;
      case ';':
        v58 = *(_QWORD *)(v4 - 64);
        v96 = 0;
        v97 = &v95;
        v59 = sub_995B10(&v96, v58);
        v60 = *(_QWORD *)(v4 - 32);
        if ( v59 && v60 )
        {
          *v97 = v60;
          return sub_1131D50(v95, v6 ^ 1u, v94, v23);
        }
        if ( (unsigned __int8)sub_995B10(&v96, v60) )
        {
          v74 = *(_QWORD *)(v4 - 64);
          if ( v74 )
          {
            *v97 = v74;
            return sub_1131D50(v95, v6 ^ 1u, v94, v23);
          }
        }
        v75 = *(_BYTE *)v4;
        if ( v6 )
        {
          v96 = &v95;
          v97 = 0;
          v98[0] = &v95;
          if ( v75 != 59 )
            return 0;
          v76 = *(_QWORD *)(v4 - 32);
          if ( *(_QWORD *)(v4 - 64) )
          {
            v77 = *(unsigned __int8 **)(v4 - 32);
            v95 = *(_QWORD *)(v4 - 64);
            result = sub_10B14D0(&v97, 15, v77);
            if ( result )
              return result;
            v76 = *(_QWORD *)(v4 - 32);
          }
          if ( v76 )
          {
            *v96 = v76;
            return sub_10B14D0(&v97, 15, *(unsigned __int8 **)(v4 - 64));
          }
          return 0;
        }
        v96 = &v95;
        v97 = &v95;
        v98[0] = 0;
        if ( v75 != 59 )
          return v6;
        v78 = *(_QWORD *)(v4 - 64);
        v79 = *(_BYTE **)(v4 - 32);
        if ( !v78 )
          goto LABEL_163;
        v95 = *(_QWORD *)(v4 - 64);
        if ( *v79 != 42 )
          goto LABEL_155;
        v81 = *((_QWORD *)v79 - 8);
        if ( !v81 || v78 != v81 )
          goto LABEL_155;
        result = sub_995B10(v98, *((_QWORD *)v79 - 4));
        if ( !result )
        {
          v79 = *(_BYTE **)(v4 - 32);
LABEL_163:
          if ( !v79 )
            return v6;
LABEL_155:
          *v96 = (__int64)v79;
          v80 = *(_BYTE **)(v4 - 64);
          if ( *v80 == 42 && *((_QWORD *)v80 - 8) == *v97 )
            return sub_995B10(v98, *((_QWORD *)v80 - 4));
          return v6;
        }
        return result;
      case 'U':
        v63 = *(_QWORD *)(v4 - 32);
        if ( !v63
          || *(_BYTE *)v63
          || *(_QWORD *)(v63 + 24) != *(_QWORD *)(v4 + 80)
          || (*(_BYTE *)(v63 + 33) & 0x20) == 0 )
        {
          return 0;
        }
        v64 = sub_987FE0(v4);
        if ( v64 > 0x14A )
        {
          if ( v64 - 365 > 1 )
            return 0;
        }
        else if ( v64 <= 0x148 )
        {
          if ( v64 != 14 )
            return 0;
          a3 = (__int64)v94;
          a4 = v23;
          a2 = v6 ^ 1;
          a1 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
          continue;
        }
        if ( !(unsigned __int8)sub_1131D50(
                                 *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))),
                                 v6,
                                 v94,
                                 v23) )
          return 0;
        a4 = v23;
        a3 = (__int64)v94;
        a2 = v6;
        a1 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
        continue;
      case 'V':
        v61 = sub_986520(v4);
        if ( !(unsigned __int8)sub_1131D50(*(_QWORD *)(v61 + 32), v6, v94, v23) )
          return 0;
        v62 = sub_986520(v4);
        a3 = (__int64)v94;
        a4 = v23;
        a2 = v6;
        a1 = *(_QWORD *)(v62 + 64);
        continue;
      default:
        return 0;
    }
  }
}
