// Function: sub_33FA050
// Address: 0x33fa050
//
unsigned __int8 *__fastcall sub_33FA050(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        unsigned __int8 *a7,
        __int64 a8)
{
  unsigned int v9; // r12d
  __int64 v11; // r11
  __int64 v12; // r10
  unsigned __int8 *result; // rax
  __int64 v14; // r10
  __int16 v15; // dx
  int v16; // r9d
  __int64 v17; // r10
  __int16 v18; // dx
  int v19; // r9d
  __m128i *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  int v23; // r14d
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // r8
  int v26; // r15d
  unsigned __int8 *v27; // rsi
  unsigned __int8 *v28; // rax
  __int64 v29; // r10
  __int64 v30; // r10
  __int64 v31; // rax
  __int16 v32; // dx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rcx
  _QWORD *v38; // rax
  __int64 v39; // rax
  __int64 v40; // r10
  int v41; // r9d
  __int64 v42; // r12
  int v43; // edx
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 v49; // rsi
  unsigned int v50; // eax
  __int64 v51; // r10
  int v52; // r9d
  __int64 v53; // r10
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned int v57; // eax
  __int64 v58; // r10
  __int64 v59; // r10
  __int16 v60; // dx
  __int64 v61; // rcx
  unsigned __int64 v62; // rax
  __int64 v63; // r10
  __int16 v64; // dx
  int v65; // r9d
  __int64 *v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rdi
  int v70; // r9d
  __int64 v71; // rax
  __int64 v72; // rax
  __int16 v73; // dx
  __int64 v74; // rax
  int v75; // eax
  __int64 v76; // rdx
  unsigned int v77; // eax
  __int64 v78; // rdx
  int v79; // r9d
  int v80; // ecx
  int v81; // r8d
  unsigned int *v82; // r14
  int v83; // r15d
  __int64 v84; // rax
  __int64 v85; // rdx
  char v86; // al
  __int64 v87; // [rsp-8h] [rbp-128h]
  __int64 v88; // [rsp+0h] [rbp-120h]
  __int64 v89; // [rsp+8h] [rbp-118h]
  unsigned int v90; // [rsp+8h] [rbp-118h]
  __int64 v92; // [rsp+10h] [rbp-110h]
  __int64 v93; // [rsp+10h] [rbp-110h]
  int v94; // [rsp+18h] [rbp-108h]
  unsigned __int8 *v95; // [rsp+18h] [rbp-108h]
  __int64 v96; // [rsp+18h] [rbp-108h]
  __int64 v97; // [rsp+18h] [rbp-108h]
  __int64 v98; // [rsp+18h] [rbp-108h]
  unsigned __int64 v99; // [rsp+18h] [rbp-108h]
  int v100; // [rsp+18h] [rbp-108h]
  __int64 v101; // [rsp+20h] [rbp-100h] BYREF
  __int64 v102; // [rsp+28h] [rbp-F8h]
  __int64 *v103; // [rsp+38h] [rbp-E8h] BYREF
  unsigned __int64 v104; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+48h] [rbp-D8h]
  __int64 v106; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v107; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v108; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v109; // [rsp+68h] [rbp-B8h]
  _BYTE v110[176]; // [rsp+70h] [rbp-B0h] BYREF

  v9 = a2;
  v101 = a4;
  v102 = a5;
  v11 = (__int64)a7;
  v12 = (unsigned int)a8;
  switch ( (int)a2 )
  {
    case 170:
    case 189:
    case 197:
    case 198:
    case 199:
    case 200:
    case 201:
    case 203:
    case 204:
    case 213:
    case 214:
    case 215:
    case 216:
    case 220:
    case 221:
    case 226:
    case 227:
    case 233:
    case 234:
    case 236:
    case 237:
    case 240:
    case 241:
    case 244:
    case 245:
    case 268:
    case 269:
    case 274:
      v108 = a7;
      v109 = a8;
      result = (unsigned __int8 *)sub_3402EA0(a1, a2, a3, v101, v102, 0, (__int64)&v108, 1);
      a2 = v87;
      v11 = (__int64)a7;
      v12 = (unsigned int)a8;
      if ( !result )
        goto LABEL_3;
      return result;
    default:
LABEL_3:
      v94 = *(_DWORD *)(v11 + 24);
      if ( v9 > 0xF5 )
      {
        if ( v9 > 0x185 )
        {
          if ( v9 != 390 )
            goto LABEL_14;
        }
        else
        {
          if ( v9 > 0x183 )
          {
            v63 = *(_QWORD *)(v11 + 48) + 16 * v12;
            v64 = *(_WORD *)v63;
            v109 = *(_QWORD *)(v63 + 8);
            LOWORD(v108) = v64;
            if ( sub_3281100((unsigned __int16 *)&v108, a2) == 2 )
              return (unsigned __int8 *)sub_33FAF80(a1, 385, a3, v101, v102, v65);
            goto LABEL_14;
          }
          if ( v9 == 382 )
          {
            v14 = *(_QWORD *)(v11 + 48) + 16 * v12;
            v15 = *(_WORD *)v14;
            v109 = *(_QWORD *)(v14 + 8);
            LOWORD(v108) = v15;
            if ( sub_3281100((unsigned __int16 *)&v108, a2) == 2 )
              return (unsigned __int8 *)sub_33FAF80(a1, 386, a3, v101, v102, v16);
            goto LABEL_14;
          }
          if ( v9 != 387 )
            goto LABEL_14;
        }
        v17 = *(_QWORD *)(v11 + 48) + 16 * v12;
        v18 = *(_WORD *)v17;
        v109 = *(_QWORD *)(v17 + 8);
        LOWORD(v108) = v18;
        if ( sub_3281100((unsigned __int16 *)&v108, a2) == 2 )
          return (unsigned __int8 *)sub_33FAF80(a1, 384, a3, v101, v102, v19);
        goto LABEL_14;
      }
      if ( v9 > 0x9B )
      {
        switch ( v9 )
        {
          case 0x9Cu:
            v108 = a7;
            v109 = a8;
            result = (unsigned __int8 *)sub_33F2070(v101, v102, (char *)&v108, 1, (_QWORD *)a1);
            if ( !result )
              goto LABEL_14;
            return result;
          case 0x9Fu:
            return a7;
          case 0xA7u:
            if ( v94 == 51 )
              return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
            if ( v94 == 158 )
            {
              v34 = *(_QWORD *)(v11 + 40);
              v35 = *(_QWORD *)(v34 + 40);
              v36 = *(_DWORD *)(v35 + 24);
              if ( v36 == 11 || v36 == 35 )
              {
                v37 = *(_QWORD *)(v35 + 96);
                v38 = *(_QWORD **)(v37 + 24);
                if ( *(_DWORD *)(v37 + 32) > 0x40u )
                  v38 = (_QWORD *)*v38;
                if ( !v38 )
                {
                  v39 = *(_QWORD *)(*(_QWORD *)v34 + 48LL) + 16LL * *(unsigned int *)(v34 + 8);
                  if ( *(_WORD *)v39 == (_WORD)v101 && (*(_QWORD *)(v39 + 8) == v102 || *(_WORD *)v39) )
                    return *(unsigned __int8 **)v34;
                }
              }
            }
            goto LABEL_14;
          case 0xBDu:
            goto LABEL_48;
          case 0xC5u:
            if ( v94 == 51 )
              return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
            if ( v94 == 197 )
              goto LABEL_105;
            goto LABEL_14;
          case 0xC6u:
          case 0xC7u:
            v30 = 16 * v12;
            v96 = v11;
            v31 = *(_QWORD *)(v11 + 48) + v30;
            v89 = v30;
            v32 = *(_WORD *)v31;
            v33 = *(_QWORD *)(v31 + 8);
            LOWORD(v108) = v32;
            v109 = v33;
            if ( sub_3281100((unsigned __int16 *)&v108, a2) != 2 )
              goto LABEL_14;
            return (unsigned __int8 *)sub_34074A0(
                                        a1,
                                        a3,
                                        a7,
                                        a8,
                                        *(unsigned __int16 *)(*(_QWORD *)(v96 + 48) + v89),
                                        *(_QWORD *)(*(_QWORD *)(v96 + 48) + v89 + 8));
          case 0xC8u:
            v59 = *(_QWORD *)(v11 + 48) + 16 * v12;
            v60 = *(_WORD *)v59;
            v109 = *(_QWORD *)(v59 + 8);
            LOWORD(v108) = v60;
            if ( sub_3281100((unsigned __int16 *)&v108, a2) == 2 )
              return a7;
            goto LABEL_14;
          case 0xC9u:
          case 0xE2u:
          case 0xE3u:
            goto LABEL_37;
          case 0xD5u:
            v40 = *(_QWORD *)(v11 + 48) + 16 * v12;
            if ( *(_WORD *)v40 == (_WORD)v101 && (*(_QWORD *)(v40 + 8) == v102 || *(_WORD *)v40) )
              return a7;
            if ( (unsigned int)(v94 - 213) <= 1 )
            {
              v41 = 0;
              if ( v94 == 214 )
              {
                v41 = *(_DWORD *)(v11 + 28) & 0x10;
                if ( v41 )
                  v41 = 16;
              }
              v42 = sub_33FA050(
                      a1,
                      v94,
                      a3,
                      v101,
                      v102,
                      v41,
                      **(_QWORD **)(v11 + 40),
                      *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8LL));
              sub_33F9B80(a1, (__int64)a7, a8, v42, v43, 0, 0, 1);
              return (unsigned __int8 *)v42;
            }
LABEL_48:
            if ( v94 != 51 )
              goto LABEL_14;
            return (unsigned __int8 *)sub_3400BD0(a1, 0, a3, v101, v102, 0, 0);
          case 0xD6u:
            v44 = *(_QWORD *)(v11 + 48) + 16LL * (unsigned int)v12;
            if ( *(_WORD *)v44 == (_WORD)v101 && (*(_QWORD *)(v44 + 8) == v102 || (_WORD)v101) )
              return a7;
            switch ( v94 )
            {
              case 214:
                v70 = 16;
                if ( (*(_DWORD *)(v11 + 28) & 0x10) == 0 )
                  v70 = 0;
                return (unsigned __int8 *)sub_33FA050(
                                            a1,
                                            214,
                                            a3,
                                            v101,
                                            v102,
                                            v70,
                                            **(_QWORD **)(v11 + 40),
                                            *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8LL));
              case 51:
                return (unsigned __int8 *)sub_3400BD0(a1, 0, a3, v101, v102, 0, 0);
              case 216:
                v45 = *(__int64 **)(v11 + 40);
                v46 = *v45;
                v47 = v45[1];
                v97 = *v45;
                v48 = *(_QWORD *)(*v45 + 48) + 16LL * *((unsigned int *)v45 + 2);
                if ( *(_WORD *)v48 == (_WORD)v101 && (*(_QWORD *)(v48 + 8) == v102 || (_WORD)v101) )
                {
                  v88 = v47;
                  if ( *(_DWORD *)(v46 + 24) != 186 )
                  {
                    v49 = (unsigned int)v12;
                    v90 = sub_33C9580(v11, v12);
                    v50 = sub_32844A0((unsigned __int16 *)&v101, v49);
                    sub_10FDA80((__int64)&v108, v50, v90);
                    if ( (unsigned __int8)sub_33DD210(a1, v97, v88, (__int64)&v108, 0) )
                    {
                      sub_33F9B80(a1, (__int64)a7, a8, v97, v88, 0, 0, 1);
                      sub_969240((__int64 *)&v108);
                      return (unsigned __int8 *)v97;
                    }
                    sub_969240((__int64 *)&v108);
                  }
                }
                break;
            }
            goto LABEL_14;
          case 0xD7u:
            v51 = *(_QWORD *)(v11 + 48) + 16 * v12;
            if ( (_WORD)v101 == *(_WORD *)v51 && (*(_QWORD *)(v51 + 8) == v102 || (_WORD)v101) )
              return a7;
            if ( (unsigned int)(v94 - 213) <= 2 )
            {
              v52 = 0;
              if ( v94 == 214 )
              {
                v52 = *(_DWORD *)(v11 + 28) & 0x10;
                if ( v52 )
                  v52 = 16;
              }
              return (unsigned __int8 *)sub_33FA050(
                                          a1,
                                          v94,
                                          a3,
                                          v101,
                                          v102,
                                          v52,
                                          **(_QWORD **)(v11 + 40),
                                          *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8LL));
            }
            if ( v94 == 51 )
              return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
            if ( v94 == 216 )
            {
              v66 = *(__int64 **)(v11 + 40);
              v67 = *v66;
              v68 = v66[1];
              v69 = *(_QWORD *)(*v66 + 48) + 16LL * *((unsigned int *)v66 + 2);
              if ( (_WORD)v101 == *(_WORD *)v69 && (*(_QWORD *)(v69 + 8) == v102 || (_WORD)v101) )
              {
                v98 = v67;
                sub_33F9B80(a1, (__int64)a7, a8, v67, v68, 0, 0, 1);
                return (unsigned __int8 *)v98;
              }
            }
            goto LABEL_14;
          case 0xD8u:
            v53 = *(_QWORD *)(v11 + 48) + 16 * v12;
            if ( *(_WORD *)v53 == (_WORD)v101 && (*(_QWORD *)(v53 + 8) == v102 || *(_WORD *)v53) )
              return a7;
            if ( v94 == 216 )
            {
              v80 = v101;
              v81 = v102;
            }
            else
            {
              if ( (unsigned int)(v94 - 213) > 2 )
              {
                if ( v94 == 51 )
                  return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
                if ( v94 == 373 && !*(_BYTE *)(a1 + 762) )
                {
                  v54 = *(_QWORD *)(**(_QWORD **)(v11 + 40) + 96LL);
                  v55 = sub_2D5B750((unsigned __int16 *)&v101);
                  v107 = v56;
                  v106 = v55;
                  v57 = sub_CA1930(&v106);
                  sub_C44740((__int64)&v108, (char **)(v54 + 24), v57);
                  v92 = sub_3401900(a1, a3, (unsigned int)v101, v102, &v108, 1);
                  sub_969240((__int64 *)&v108);
                  return (unsigned __int8 *)v92;
                }
                goto LABEL_14;
              }
              v93 = v11;
              v72 = *(_QWORD *)(**(_QWORD **)(v11 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v11 + 40) + 8LL);
              v73 = *(_WORD *)v72;
              v74 = *(_QWORD *)(v72 + 8);
              LOWORD(v106) = v73;
              v107 = v74;
              LOWORD(v75) = sub_3281100((unsigned __int16 *)&v106, a2);
              LODWORD(v108) = v75;
              v109 = v76;
              LOWORD(v77) = sub_3281100((unsigned __int16 *)&v101, a2);
              if ( sub_3280B30((__int64)&v108, v77, v78) )
                return (unsigned __int8 *)sub_33FAF80(a1, v94, a3, v101, v102, v79);
              v82 = *(unsigned int **)(v93 + 40);
              v83 = v102;
              v100 = v101;
              v84 = *(_QWORD *)(*(_QWORD *)v82 + 48LL) + 16LL * v82[2];
              v85 = *(_QWORD *)(v84 + 8);
              LOWORD(v84) = *(_WORD *)v84;
              v109 = v85;
              LOWORD(v108) = v84;
              v86 = sub_3280A00((__int64)&v108, (unsigned int)v101, v102);
              v80 = v100;
              if ( !v86 )
                return *(unsigned __int8 **)v82;
              v81 = v83;
            }
            return (unsigned __int8 *)sub_33FAF80(a1, 216, a3, v80, v81, a6);
          case 0xDCu:
          case 0xDDu:
            if ( v94 != 51 )
              goto LABEL_14;
            return (unsigned __int8 *)sub_33FE730(a1, a3, (unsigned int)v101, v102, 0, 0.0);
          case 0xE6u:
            BUG();
          case 0xE9u:
            v29 = *(_QWORD *)(v11 + 48) + 16 * v12;
            if ( (_WORD)v101 != *(_WORD *)v29 || *(_QWORD *)(v29 + 8) != v102 && !*(_WORD *)v29 )
              goto LABEL_37;
            return a7;
          case 0xEAu:
            v58 = *(_QWORD *)(v11 + 48) + 16 * v12;
            if ( (_WORD)v101 == *(_WORD *)v58 && ((_WORD)v101 || v102 == *(_QWORD *)(v58 + 8)) )
              return a7;
            if ( v94 == 234 )
              return (unsigned __int8 *)sub_33FAF80(a1, 234, a3, v101, v102, a6);
LABEL_37:
            if ( v94 != 51 )
              goto LABEL_14;
            return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
          case 0xF4u:
            if ( v94 == 51 )
              return (unsigned __int8 *)sub_3288990(a1, (unsigned int)v101, v102);
            if ( v94 != 244 )
              goto LABEL_14;
LABEL_105:
            v34 = *(_QWORD *)(v11 + 40);
            return *(unsigned __int8 **)v34;
          case 0xF5u:
            if ( v94 != 244 )
              goto LABEL_14;
            return (unsigned __int8 *)sub_33FAF80(a1, 245, a3, v101, v102, a6);
          default:
            goto LABEL_14;
        }
      }
      if ( v9 != 52 )
      {
        if ( v9 != 55 && v9 != 2 )
          goto LABEL_14;
        return a7;
      }
      if ( (unsigned __int8)sub_33DE850((_QWORD **)a1, (__int64)a7, a8, 0, 1u) )
        return a7;
LABEL_14:
      v20 = sub_33ED250(a1, (unsigned int)v101, v102);
      v106 = (__int64)a7;
      v104 = (unsigned __int64)v20;
      v105 = v21;
      v107 = a8;
      if ( (_WORD)v101 == 262 )
      {
        v22 = *(_QWORD *)a3;
        v23 = *(_DWORD *)(a3 + 8);
        v108 = (unsigned __int8 *)v22;
        if ( v22 )
          sub_B96E90((__int64)&v108, v22, 1);
        v24 = *(_QWORD *)(a1 + 416);
        v25 = v104;
        v26 = v105;
        if ( v24 )
        {
          *(_QWORD *)(a1 + 416) = *(_QWORD *)v24;
        }
        else
        {
          v61 = *(_QWORD *)(a1 + 424);
          *(_QWORD *)(a1 + 504) += 120LL;
          v62 = (v61 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          if ( *(_QWORD *)(a1 + 432) >= v62 + 120 && v61 )
          {
            *(_QWORD *)(a1 + 424) = v62 + 120;
            if ( !v62 )
            {
              if ( v108 )
                sub_B91220((__int64)&v108, (__int64)v108);
              goto LABEL_22;
            }
            v24 = (v61 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          }
          else
          {
            v99 = v25;
            v71 = sub_9D1E70(a1 + 424, 120, 120, 3);
            v25 = v99;
            v24 = v71;
          }
        }
        *(_QWORD *)v24 = 0;
        *(_QWORD *)(v24 + 8) = 0;
        *(_QWORD *)(v24 + 16) = 0;
        *(_DWORD *)(v24 + 24) = v9;
        *(_DWORD *)(v24 + 28) = 0;
        *(_WORD *)(v24 + 34) = -1;
        *(_DWORD *)(v24 + 36) = -1;
        *(_QWORD *)(v24 + 40) = 0;
        *(_QWORD *)(v24 + 48) = v25;
        *(_QWORD *)(v24 + 56) = 0;
        *(_DWORD *)(v24 + 64) = 0;
        *(_DWORD *)(v24 + 68) = v26;
        *(_DWORD *)(v24 + 72) = v23;
        v27 = v108;
        *(_QWORD *)(v24 + 80) = v108;
        if ( v27 )
          sub_B976B0((__int64)&v108, v27, v24 + 80);
        *(_QWORD *)(v24 + 88) = 0xFFFFFFFFLL;
        *(_WORD *)(v24 + 32) = 0;
LABEL_22:
        sub_33E4EC0(a1, v24, (__int64)&v106, 1);
LABEL_23:
        sub_33CC420(a1, v24);
        return (unsigned __int8 *)v24;
      }
      v108 = v110;
      v109 = 0x2000000000LL;
      sub_33C9670((__int64)&v108, v9, v104, (unsigned __int64 *)&v106, 1, (__int64)&v108);
      v103 = 0;
      v28 = (unsigned __int8 *)sub_33CCCF0(a1, (__int64)&v108, a3, (__int64 *)&v103);
      if ( !v28 )
      {
        v24 = sub_33E6540((_QWORD *)a1, v9, *(_DWORD *)(a3 + 8), (__int64 *)a3, (__int64 *)&v104);
        *(_DWORD *)(v24 + 28) = a6;
        sub_33E4EC0(a1, v24, (__int64)&v106, 1);
        sub_C657C0((__int64 *)(a1 + 520), (__int64 *)v24, v103, (__int64)off_4A367D0);
        if ( v108 != v110 )
          _libc_free((unsigned __int64)v108);
        goto LABEL_23;
      }
      v95 = v28;
      sub_33D00A0((__int64)v28, a6);
      result = v95;
      if ( v108 != v110 )
      {
        _libc_free((unsigned __int64)v108);
        return v95;
      }
      return result;
  }
}
