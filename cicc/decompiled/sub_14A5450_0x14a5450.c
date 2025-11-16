// Function: sub_14A5450
// Address: 0x14a5450
//
__int64 __fastcall sub_14A5450(__int64 **a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // r11
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r14
  char *v11; // rdx
  int v12; // eax
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 result; // rax
  void *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // rbx
  _QWORD *v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // r14
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // r14
  char *v30; // rdx
  int v31; // ecx
  char *v32; // r8
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // r15
  __int64 *v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 (*v40)(); // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rbx
  int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  int v49; // edx
  _QWORD *v50; // r14
  __int64 v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // rbx
  unsigned __int64 v54; // r15
  char *v55; // rax
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rsi
  unsigned __int8 v60; // cl
  __int64 v61; // rax
  char *v62; // rdi
  __int64 v63; // r14
  char v64; // bl
  __int64 v65; // rdi
  char v66; // bl
  __int64 v67; // rdi
  char v68; // bl
  __int64 v69; // rdi
  char v70; // bl
  __int64 v71; // rdi
  char v72; // bl
  __int64 v73; // rdi
  __int64 v74; // rbx
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned __int8 v77; // r8
  unsigned __int8 v78; // cl
  __int64 v79; // rdi
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // rcx
  unsigned int v83; // eax
  __int64 v84; // rdx
  char *v85; // rdi
  _BYTE *v86; // r8
  size_t v87; // rdx
  int v88; // eax
  int v89; // ebx
  __int64 v90; // rdi
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 v94; // rbx
  __int64 v95; // rax
  _BYTE *v96; // [rsp+0h] [rbp-1A0h]
  __int64 v97; // [rsp+8h] [rbp-198h]
  __int64 v98; // [rsp+10h] [rbp-190h]
  __int64 v99; // [rsp+10h] [rbp-190h]
  int v100; // [rsp+10h] [rbp-190h]
  __int64 v101; // [rsp+18h] [rbp-188h]
  __int64 v102; // [rsp+20h] [rbp-180h]
  int v103; // [rsp+20h] [rbp-180h]
  int v104; // [rsp+20h] [rbp-180h]
  unsigned int v105; // [rsp+20h] [rbp-180h]
  unsigned int v106; // [rsp+28h] [rbp-178h]
  unsigned int v107; // [rsp+28h] [rbp-178h]
  unsigned int v108; // [rsp+28h] [rbp-178h]
  unsigned int v109[6]; // [rsp+30h] [rbp-170h] BYREF
  int v110; // [rsp+48h] [rbp-158h]
  char v111; // [rsp+50h] [rbp-150h]
  int v112; // [rsp+60h] [rbp-140h] BYREF
  __int64 v113; // [rsp+68h] [rbp-138h]
  __int64 v114; // [rsp+70h] [rbp-130h]
  int v115; // [rsp+78h] [rbp-128h]
  char v116; // [rsp+80h] [rbp-120h]
  void *s2; // [rsp+90h] [rbp-110h] BYREF
  __int64 v118; // [rsp+98h] [rbp-108h]
  _BYTE v119[64]; // [rsp+A0h] [rbp-100h] BYREF
  void *s1; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v121; // [rsp+E8h] [rbp-B8h]
  char s[8]; // [rsp+F0h] [rbp-B0h] BYREF
  int v123; // [rsp+F8h] [rbp-A8h]
  char v124; // [rsp+100h] [rbp-A0h]

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x19:
    case 0x1A:
    case 0x4D:
      return sub_14A3410((__int64)a1);
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
      v4 = a2;
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v5 = *(__int64 **)(a2 - 8);
      else
        v5 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      sub_14A0F00(*v5, &v112);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v6 = *(_QWORD *)(a2 - 8);
      else
        v6 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      sub_14A0F00(*(_QWORD *)(v6 + 24), &s2);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v7 = *(__int64 **)(a2 - 8);
        v4 = (__int64)&v7[3 * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)];
      }
      else
      {
        v7 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      }
      v8 = v4 - (_QWORD)v7;
      v121 = 0x200000000LL;
      s1 = s;
      v9 = 0xAAAAAAAAAAAAAAABLL * ((v4 - (__int64)v7) >> 3);
      v10 = v9;
      if ( (unsigned __int64)v8 > 0x30 )
      {
        v98 = v8;
        v103 = v9;
        sub_16CD150(&s1, s, v9, 8);
        v12 = v121;
        LODWORD(v9) = v103;
        v8 = v98;
        v11 = (char *)s1 + 8 * (unsigned int)v121;
      }
      else
      {
        v11 = s;
        v12 = 0;
      }
      if ( v8 > 0 )
      {
        v13 = v7;
        do
        {
          v14 = *v13;
          v11 += 8;
          v13 += 3;
          *((_QWORD *)v11 - 1) = v14;
          --v10;
        }
        while ( v10 );
        v12 = v121;
      }
      LODWORD(v121) = v9 + v12;
      result = sub_14A3350((__int64)a1);
      v16 = s1;
      if ( s1 == s )
        return result;
      goto LABEL_47;
    case 0x36:
    case 0x37:
      return sub_14A34A0((__int64)a1);
    case 0x38:
      v24 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v25 = *(__int64 **)(a2 - 8);
        v26 = (__int64)&v25[v24];
      }
      else
      {
        v25 = (__int64 *)(a2 - v24 * 8);
        v26 = a2;
      }
      v27 = v26 - (_QWORD)v25;
      v121 = 0x400000000LL;
      s1 = s;
      v28 = 0xAAAAAAAAAAAAAAABLL * (v27 >> 3);
      v29 = v28;
      if ( (unsigned __int64)v27 > 0x60 )
      {
        v102 = v27;
        sub_16CD150(&s1, s, 0xAAAAAAAAAAAAAAABLL * (v27 >> 3), 8);
        v32 = (char *)s1;
        v31 = v121;
        v27 = v102;
        v30 = (char *)s1 + 8 * (unsigned int)v121;
      }
      else
      {
        v30 = s;
        v31 = 0;
        v32 = s;
      }
      if ( v27 > 0 )
      {
        v33 = v25;
        do
        {
          v34 = *v33;
          v30 += 8;
          v33 += 3;
          *((_QWORD *)v30 - 1) = v34;
          --v29;
        }
        while ( v29 );
        v32 = (char *)s1;
        v31 = v121;
      }
      LODWORD(v121) = v31 + v28;
      result = sub_14A5330(a1, a2, (__int64)v32, (unsigned int)(v31 + v28));
      v16 = s1;
      if ( s1 != s )
        goto LABEL_47;
      return result;
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x45:
    case 0x46:
    case 0x47:
    case 0x48:
      return sub_14A33B0((__int64)a1);
    case 0x4B:
    case 0x4C:
    case 0x4F:
      return sub_14A3440((__int64)a1);
    case 0x4E:
      v41 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v41 + 16) || (*(_BYTE *)(v41 + 33) & 0x20) == 0 )
        return 0xFFFFFFFFLL;
      if ( *(char *)(a2 + 23) >= 0 )
        goto LABEL_75;
      v42 = sub_1648A40(a2);
      v44 = v42 + v43;
      if ( *(char *)(a2 + 23) >= 0 )
      {
        if ( (unsigned int)(v44 >> 4) )
LABEL_191:
          BUG();
LABEL_75:
        v48 = -24;
        goto LABEL_64;
      }
      if ( !(unsigned int)((v44 - sub_1648A40(a2)) >> 4) )
        goto LABEL_75;
      if ( *(char *)(a2 + 23) >= 0 )
        goto LABEL_191;
      v45 = *(_DWORD *)(sub_1648A40(a2) + 8);
      if ( *(char *)(a2 + 23) >= 0 )
        BUG();
      v46 = sub_1648A40(a2);
      v48 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v46 + v47 - 4) - v45);
LABEL_64:
      v49 = *(_DWORD *)(a2 + 20);
      v50 = (_QWORD *)(a2 + v48);
      s1 = s;
      v121 = 0x400000000LL;
      v51 = 24LL * (v49 & 0xFFFFFFF);
      v52 = v51 + v48;
      v53 = (_QWORD *)(a2 - v51);
      v54 = 0xAAAAAAAAAAAAAAABLL * (v52 >> 3);
      if ( (unsigned __int64)v52 > 0x60 )
      {
        sub_16CD150(&s1, s, 0xAAAAAAAAAAAAAAABLL * (v52 >> 3), 8);
        v56 = v121;
        v55 = (char *)s1 + 8 * (unsigned int)v121;
      }
      else
      {
        v55 = s;
        v56 = 0;
      }
      if ( v50 != v53 )
      {
        do
        {
          if ( v55 )
            *(_QWORD *)v55 = *v53;
          v53 += 3;
          v55 += 8;
        }
        while ( v50 != v53 );
        v56 = v121;
      }
      LODWORD(v121) = v54 + v56;
      if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) )
        BUG();
      result = sub_14A3590((__int64)a1);
      v16 = s1;
      if ( s1 != s )
      {
LABEL_47:
        v107 = result;
        _libc_free((unsigned __int64)v16);
        return v107;
      }
      return result;
    case 0x53:
      if ( !byte_4F9D480 )
        return sub_14A3470((__int64)a1);
      v18 = *(_QWORD *)(a2 - 24);
      v19 = *(__int64 **)(a2 - 48);
      if ( *(_BYTE *)(v18 + 16) != 13 )
        goto LABEL_83;
      v20 = *(_QWORD **)(v18 + 24);
      if ( *(_DWORD *)(v18 + 32) > 0x40u )
        v20 = (_QWORD *)*v20;
      if ( (_DWORD)v20 || *((_BYTE *)v19 + 16) <= 0x17u )
        goto LABEL_83;
      sub_14A4410((__int64)v109, *(_QWORD *)(a2 - 48));
      v21 = *(__int64 **)(a2 - 48);
      if ( !v111 )
        goto LABEL_36;
      v22 = *v21;
      v23 = *(_QWORD *)(*v21 + 32);
      v106 = v23;
      if ( !(_DWORD)v23 || ((unsigned int)v23 & ((_DWORD)v23 - 1)) != 0 )
        goto LABEL_36;
      v61 = (unsigned int)v23;
      v121 = 0x2000000000LL;
      v62 = s;
      s1 = s;
      if ( (unsigned int)v23 > 0x20 )
      {
        v100 = v23;
        v101 = (unsigned int)v23;
        sub_16CD150(&s1, s, (unsigned int)v23, 4);
        v62 = (char *)s1;
        LODWORD(v23) = v100;
        v61 = v101;
      }
      LODWORD(v121) = v23;
      if ( 4 * v61 )
      {
        v104 = v23;
        memset(v62, 0, 4 * v61);
        LODWORD(v23) = v104;
      }
      if ( (_DWORD)v23 == 1 )
        goto LABEL_173;
      v99 = v22;
      v63 = (__int64)v19;
      v105 = 1;
      while ( 2 )
      {
        sub_14A4410((__int64)&v112, v63);
        if ( !v116 || v115 != v110 || v112 != v109[0] )
          goto LABEL_106;
        v82 = v113;
        v63 = v114;
        if ( *(_BYTE *)(v113 + 16) != 85 )
        {
          if ( *(_BYTE *)(v114 + 16) != 85 )
            goto LABEL_106;
          v82 = v114;
          v63 = v113;
        }
        if ( v63 != *(_QWORD *)(v82 - 72) )
          goto LABEL_106;
        if ( v105 )
        {
          v83 = v105;
          v84 = 0;
          do
          {
            *(_DWORD *)((char *)s1 + v84) = v83++;
            v84 += 4;
          }
          while ( 2 * v105 != v83 );
        }
        v85 = (char *)s1 + 4 * v105;
        if ( v85 != (char *)s1 + 4 * (unsigned int)v121 )
        {
          v97 = v82;
          memset(v85, 255, 4 * ((unsigned int)v121 - (unsigned __int64)v105));
          v82 = v97;
        }
        s2 = v119;
        v118 = 0x1000000000LL;
        sub_15FAA20(*(_QWORD *)(v82 - 24), &s2);
        if ( (unsigned int)v121 == (unsigned __int64)(unsigned int)v118 )
        {
          v86 = s2;
          v87 = 4LL * (unsigned int)v121;
          if ( !v87 || (v96 = s2, v88 = memcmp(s1, s2, v87), v86 = v96, !v88) )
          {
            if ( *(_BYTE *)(v63 + 16) <= 0x17u )
              v63 = 0;
            v106 >>= 1;
            v105 *= 2;
            if ( v86 != v119 )
              _libc_free((unsigned __int64)v86);
            if ( v106 != 1 )
            {
              if ( !v63 )
                goto LABEL_106;
              continue;
            }
            v22 = v99;
LABEL_173:
            v59 = v109[0];
            v89 = v110;
            if ( s1 != s )
            {
              v108 = v109[0];
              _libc_free((unsigned __int64)s1);
              v59 = v108;
            }
            if ( v89 == 2 )
            {
              v93 = *(_QWORD *)v22;
              if ( *(_BYTE *)(v22 + 8) == 16 )
              {
                v94 = *(_QWORD *)(v22 + 32);
                v95 = sub_1643320(v93);
                v76 = sub_16463B0(v95, (unsigned int)v94);
              }
              else
              {
                v76 = sub_1643320(v93);
              }
              v77 = 0;
              v78 = 0;
              return sub_14A3680((__int64 *)a1, v22, v76, v78, v77);
            }
            if ( v89 == 3 )
            {
              v90 = *(_QWORD *)v22;
              if ( *(_BYTE *)(v22 + 8) == 16 )
              {
                v91 = *(_QWORD *)(v22 + 32);
                v92 = sub_1643320(v90);
                v76 = sub_16463B0(v92, (unsigned int)v91);
              }
              else
              {
                v76 = sub_1643320(v90);
              }
              v77 = 1;
              v78 = 0;
              return sub_14A3680((__int64 *)a1, v22, v76, v78, v77);
            }
            v60 = 0;
            if ( v89 == 1 )
              return sub_14A3650((__int64 *)a1, v59, v22, v60);
LABEL_108:
            v21 = *(__int64 **)(a2 - 48);
LABEL_36:
            if ( !byte_4F9D480 )
              return sub_14A3470((__int64)a1);
            v18 = *(_QWORD *)(a2 - 24);
            v19 = v21;
LABEL_83:
            if ( *(_BYTE *)(v18 + 16) != 13 )
              return sub_14A3470((__int64)a1);
            v57 = *(_DWORD *)(v18 + 32) <= 0x40u ? *(_QWORD *)(v18 + 24) : **(_QWORD **)(v18 + 24);
            if ( (_DWORD)v57 )
              return sub_14A3470((__int64)a1);
            if ( *((_BYTE *)v19 + 16) <= 0x17u )
              return sub_14A3470((__int64)a1);
            sub_14A4410((__int64)&s1, (__int64)v19);
            if ( !v124 )
              return sub_14A3470((__int64)a1);
            v22 = *v19;
            v58 = *(_QWORD *)(*v19 + 32);
            if ( !(_DWORD)v58 )
              return sub_14A3470((__int64)a1);
            if ( ((unsigned int)v58 & ((_DWORD)v58 - 1)) != 0 )
              return sub_14A3470((__int64)a1);
            _BitScanReverse((unsigned int *)&v58, v58);
            if ( !(unsigned int)sub_14A4980((__int64)v19, 0, 31 - ((unsigned int)v58 ^ 0x1F)) )
              return sub_14A3470((__int64)a1);
            v59 = (unsigned int)s1;
            switch ( v123 )
            {
              case 2:
                v73 = *(_QWORD *)v22;
                if ( *(_BYTE *)(v22 + 8) == 16 )
                {
                  v74 = *(_QWORD *)(v22 + 32);
                  v75 = sub_1643320(v73);
                  v76 = sub_16463B0(v75, (unsigned int)v74);
                }
                else
                {
                  v76 = sub_1643320(v73);
                }
                v77 = 0;
                v78 = 1;
                break;
              case 3:
                v79 = *(_QWORD *)v22;
                if ( *(_BYTE *)(v22 + 8) == 16 )
                {
                  v80 = *(_QWORD *)(v22 + 32);
                  v81 = sub_1643320(v79);
                  v76 = sub_16463B0(v81, (unsigned int)v80);
                }
                else
                {
                  v76 = sub_1643320(v79);
                }
                v77 = 1;
                v78 = 1;
                break;
              case 1:
                v60 = 1;
                return sub_14A3650((__int64 *)a1, v59, v22, v60);
              default:
                return sub_14A3470((__int64)a1);
            }
            return sub_14A3680((__int64 *)a1, v22, v76, v78, v77);
          }
        }
        else
        {
          v86 = s2;
        }
        break;
      }
      if ( v86 != v119 )
        _libc_free((unsigned __int64)v86);
LABEL_106:
      if ( s1 != s )
        _libc_free((unsigned __int64)s1);
      goto LABEL_108;
    case 0x54:
      return sub_14A3470((__int64)a1);
    case 0x55:
      v17 = *(_QWORD *)(a2 - 24);
      if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) != *(_DWORD *)(*(_QWORD *)v17 + 32LL) )
        return 0xFFFFFFFFLL;
      s1 = s;
      v121 = 0x1000000000LL;
      sub_15FAA20(v17, &s1);
      if ( (unsigned __int8)sub_15FAB90(s1, (unsigned int)v121) )
      {
        if ( s1 != s )
          _libc_free((unsigned __int64)s1);
        return 0;
      }
      if ( s1 != s )
        _libc_free((unsigned __int64)s1);
      v35 = *(_QWORD *)(a2 - 24);
      v36 = *(_QWORD *)a2;
      if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) != *(_DWORD *)(*(_QWORD *)v35 + 32LL) )
        goto LABEL_51;
      s1 = s;
      v121 = 0x1000000000LL;
      sub_15FAA20(v35, &s1);
      v64 = sub_15FAC00(s1, (unsigned int)v121);
      if ( s1 != s )
        _libc_free((unsigned __int64)s1);
      if ( v64 )
      {
        v37 = *a1;
        v40 = *(__int64 (**)())(**a1 + 592);
        if ( v40 != sub_14A09E0 )
        {
          v38 = v36;
          v39 = 1;
          return ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _QWORD))v40)(v37, v39, v38, 0, 0);
        }
      }
      else
      {
        v65 = *(_QWORD *)(a2 - 24);
        if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) != *(_DWORD *)(*(_QWORD *)v65 + 32LL) )
          goto LABEL_51;
        s1 = s;
        v121 = 0x1000000000LL;
        sub_15FAA20(v65, &s1);
        v66 = sub_15FACD0(s1, (unsigned int)v121);
        if ( s1 != s )
          _libc_free((unsigned __int64)s1);
        if ( v66 )
        {
          v37 = *a1;
          v40 = *(__int64 (**)())(**a1 + 592);
          if ( v40 != sub_14A09E0 )
          {
            v38 = v36;
            v39 = 2;
            return ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _QWORD))v40)(v37, v39, v38, 0, 0);
          }
        }
        else
        {
          v67 = *(_QWORD *)(a2 - 24);
          if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) != *(_DWORD *)(*(_QWORD *)v67 + 32LL) )
            goto LABEL_51;
          s1 = s;
          v121 = 0x1000000000LL;
          sub_15FAA20(v67, &s1);
          v68 = sub_15FAD30(s1, (unsigned int)v121);
          if ( s1 != s )
            _libc_free((unsigned __int64)s1);
          if ( !v68 )
          {
            v69 = *(_QWORD *)(a2 - 24);
            if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) == *(_DWORD *)(*(_QWORD *)v69 + 32LL) )
            {
              s1 = s;
              v121 = 0x1000000000LL;
              sub_15FAA20(v69, &s1);
              v70 = sub_15FAC70(s1, (unsigned int)v121);
              if ( s1 != s )
                _libc_free((unsigned __int64)s1);
              if ( v70 )
                return (*(__int64 (__fastcall **)(__int64 *, _QWORD, __int64, _QWORD, _QWORD))(**a1 + 592))(
                         *a1,
                         0,
                         v36,
                         0,
                         0);
              v71 = *(_QWORD *)(a2 - 24);
              if ( *(_DWORD *)(**(_QWORD **)(a2 - 72) + 32LL) == *(_DWORD *)(*(_QWORD *)v71 + 32LL) )
              {
                s1 = s;
                v121 = 0x1000000000LL;
                sub_15FAA20(v71, &s1);
                v72 = sub_15FAB40(s1, (unsigned int)v121);
                if ( s1 != s )
                  _libc_free((unsigned __int64)s1);
                v37 = *a1;
                if ( v72 )
                  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, _QWORD, _QWORD))(*v37 + 592))(
                           v37,
                           7,
                           v36,
                           0,
                           0);
                goto LABEL_52;
              }
            }
LABEL_51:
            v37 = *a1;
LABEL_52:
            v38 = v36;
            v39 = 6;
            v40 = *(__int64 (**)())(*v37 + 592);
            return ((__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v40)(v37, v39, v38, 0, 0);
          }
          v37 = *a1;
          v40 = *(__int64 (**)())(**a1 + 592);
          if ( v40 != sub_14A09E0 )
          {
            v38 = v36;
            v39 = 3;
            return ((__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v40)(v37, v39, v38, 0, 0);
          }
        }
      }
      return 1;
    default:
      return 0xFFFFFFFFLL;
  }
}
