// Function: sub_3355E60
// Address: 0x3355e60
//
char __fastcall sub_3355E60(_QWORD *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r13
  __int64 *v7; // rax
  __int64 v8; // r14
  __int64 i; // rbx
  __int64 v10; // rdi
  __int64 v11; // rax
  char v12; // r10
  __int64 v13; // rdi
  __int64 v14; // r11
  char v15; // bl
  __int64 *v16; // rsi
  __int64 *v17; // r9
  __int64 *v18; // rdx
  char v19; // cl
  __int64 *v20; // r9
  __int64 v21; // r15
  __int64 v22; // rax
  int v23; // edx
  unsigned int *v24; // rax
  __int64 v25; // r10
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  bool v31; // zf
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v34; // r14
  unsigned int v35; // ebx
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r13
  unsigned int v38; // r12d
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // r8
  unsigned __int16 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rsi
  _QWORD *v45; // rdx
  unsigned __int16 *v46; // rbx
  __int64 v47; // r14
  unsigned __int64 v48; // rax
  __int64 v49; // r12
  __int64 v50; // r15
  unsigned int v51; // eax
  unsigned __int16 *v52; // r12
  unsigned int v53; // edx
  unsigned int v54; // esi
  __int64 v55; // r14
  __int64 v56; // r8
  _QWORD *v57; // r12
  unsigned int v58; // ebx
  __int64 v59; // rdi
  _QWORD *v60; // rsi
  _QWORD *v61; // rcx
  _QWORD *v62; // rax
  __int64 *v63; // rdx
  __int64 v64; // rdx
  int v65; // edx
  unsigned __int64 v66; // r15
  _QWORD *v67; // r11
  _QWORD *v68; // r13
  char v69; // al
  unsigned __int64 v70; // r10
  __int64 v71; // rcx
  char v72; // al
  unsigned int v73; // eax
  unsigned int v74; // edi
  __int64 v75; // rdx
  __int64 v76; // r8
  __int64 v77; // rcx
  unsigned __int64 v78; // r8
  unsigned __int64 v79; // r9
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rcx
  unsigned __int64 v85; // r8
  unsigned __int64 v86; // r9
  __int64 v87; // rdi
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  __int64 v91; // rcx
  unsigned __int64 v92; // r8
  unsigned __int64 v93; // r9
  __m128i v94; // xmm0
  _QWORD *v96; // [rsp+8h] [rbp-C8h]
  _QWORD *v97; // [rsp+10h] [rbp-C0h]
  __int64 v98; // [rsp+18h] [rbp-B8h]
  __int64 v99; // [rsp+20h] [rbp-B0h]
  _QWORD *v100; // [rsp+28h] [rbp-A8h]
  char v101; // [rsp+33h] [rbp-9Dh]
  unsigned int v102; // [rsp+34h] [rbp-9Ch]
  __int64 *v104; // [rsp+40h] [rbp-90h]
  __int64 v105; // [rsp+48h] [rbp-88h]
  __int64 v106; // [rsp+50h] [rbp-80h]
  unsigned __int16 *v107; // [rsp+58h] [rbp-78h]
  unsigned __int16 *v108; // [rsp+60h] [rbp-70h]
  __int64 v109; // [rsp+68h] [rbp-68h]
  __int64 v110; // [rsp+70h] [rbp-60h]
  __int64 v111; // [rsp+70h] [rbp-60h]
  __int64 v112; // [rsp+78h] [rbp-58h]
  __int64 v113; // [rsp+78h] [rbp-58h]
  _QWORD *v114; // [rsp+80h] [rbp-50h]
  _QWORD *v115; // [rsp+80h] [rbp-50h]
  unsigned int v116; // [rsp+88h] [rbp-48h]
  __int64 v117; // [rsp+88h] [rbp-48h]
  unsigned __int64 v118; // [rsp+88h] [rbp-48h]
  __int64 v119; // [rsp+88h] [rbp-48h]
  __m128i v120; // [rsp+90h] [rbp-40h] BYREF

  v6 = a1;
  a1[6] = a2;
  v104 = (__int64 *)a2;
  if ( !(_BYTE)qword_5038808 )
  {
    v110 = a2->m128i_i64[1];
    if ( a2->m128i_i64[0] == v110 )
      goto LABEL_3;
    v21 = a2->m128i_i64[0];
    while ( 1 )
    {
      if ( (*(_BYTE *)(v21 + 248) & 8) != 0 )
      {
        v22 = *(_QWORD *)v21;
        if ( *(_QWORD *)v21 )
        {
          if ( *(int *)(v22 + 24) < 0 )
          {
            v23 = *(_DWORD *)(v22 + 64);
            if ( !v23
              || (v24 = (unsigned int *)(*(_QWORD *)(v22 + 40) + 40LL * (unsigned int)(v23 - 1)),
                  *(_WORD *)(*(_QWORD *)(*(_QWORD *)v24 + 48LL) + 16LL * v24[2]) != 262) )
            {
              v101 = sub_3351570(v21);
              v26 = *(_QWORD *)(a1[8] + 8LL) - 40 * v25;
              v27 = *(unsigned __int16 *)(v26 + 2);
              v108 = (unsigned __int16 *)v26;
              v116 = *(unsigned __int8 *)(v26 + 4);
              v102 = v27;
              if ( v116 != v27 )
                break;
            }
          }
        }
      }
LABEL_60:
      v21 += 256;
      if ( v110 == v21 )
      {
        v6 = a1;
        v110 = a2->m128i_i64[1];
        goto LABEL_3;
      }
    }
    v109 = 0;
    while ( 1 )
    {
      if ( v27 > v116 && (v108[20 * *v108 + 22 + 3 * v108[8] + 3 * v116] & 1) != 0 )
      {
        v28 = v109;
        v29 = *(int *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v21 + 40LL) + v109) + 36LL);
        if ( (_DWORD)v29 != -1 )
        {
          v30 = v29 << 8;
          v31 = a2->m128i_i64[0] + v30 == 0;
          v32 = a2->m128i_i64[0] + v30;
          v99 = v32;
          if ( !v31 )
          {
            v33 = *(_QWORD *)(v32 + 120);
            v114 = (_QWORD *)(v33 + 16LL * *(unsigned int *)(v32 + 128));
            if ( (_QWORD *)v33 != v114 )
              break;
          }
        }
      }
LABEL_90:
      ++v116;
      v109 += 40;
      if ( v102 == v116 )
        goto LABEL_60;
      v27 = v108[1];
    }
    v34 = *(_QWORD **)(v32 + 120);
    while ( 1 )
    {
      while ( 1 )
      {
        if ( (*v34 & 6) == 0 )
        {
          v36 = *v34 & 0xFFFFFFFFFFFFFFF8LL;
          v37 = v36;
          if ( v36 != v21 )
            break;
        }
LABEL_50:
        v34 += 2;
        if ( v114 == v34 )
          goto LABEL_89;
      }
      if ( (*(_BYTE *)(v36 + 254) & 2) == 0 )
        sub_2F8F770(v36, a2, v33, v28, v36, a6);
      v38 = *(_DWORD *)(v37 + 244);
      if ( (*(_BYTE *)(v21 + 254) & 2) != 0 )
      {
        v35 = *(_DWORD *)(v21 + 244);
        if ( v38 >= v35 )
          goto LABEL_63;
LABEL_48:
        if ( (*(_BYTE *)(v37 + 254) & 2) != 0 )
        {
LABEL_49:
          if ( v35 - *(_DWORD *)(v37 + 244) <= 1 )
            goto LABEL_63;
          goto LABEL_50;
        }
LABEL_59:
        sub_2F8F770(v37, a2, v33, v28, v36, a6);
        goto LABEL_49;
      }
      sub_2F8F770(v21, a2, v33, v28, v36, a6);
      v35 = *(_DWORD *)(v21 + 244);
      if ( v38 < v35 )
      {
        if ( (*(_BYTE *)(v21 + 254) & 2) != 0 )
          goto LABEL_48;
        sub_2F8F770(v21, a2, v33, v28, v36, a6);
        v35 = *(_DWORD *)(v21 + 244);
        if ( (*(_BYTE *)(v37 + 254) & 2) != 0 )
          goto LABEL_49;
        goto LABEL_59;
      }
LABEL_63:
      while ( *(_DWORD *)(v37 + 128) == 1 )
      {
        v39 = *(_QWORD *)v37;
        v40 = *(_DWORD *)(*(_QWORD *)v37 + 24LL);
        if ( v40 != -14 )
          goto LABEL_65;
        v37 = **(_QWORD **)(v37 + 120) & 0xFFFFFFFFFFFFFFF8LL;
      }
      v39 = *(_QWORD *)v37;
      if ( !*(_QWORD *)v37 )
        goto LABEL_50;
      v40 = *(_DWORD *)(v39 + 24);
LABEL_65:
      if ( v40 >= 0 )
        goto LABEL_50;
      if ( (*(_BYTE *)(v37 + 248) & 0x40) != 0 && *(char *)(v21 + 248) < 0 )
      {
        a2 = *(__m128i **)v21;
        if ( (unsigned __int8)sub_3353030(v39, *(_QWORD *)v21, a1[8], a1[9], v36) )
          goto LABEL_50;
      }
      v33 = (unsigned int)(-*(_DWORD *)(*(_QWORD *)v37 + 24LL) - 9);
      if ( (unsigned int)v33 <= 1 || ~*(_DWORD *)(*(_QWORD *)v37 + 24LL) == 12 )
        goto LABEL_50;
      v41 = a1[11];
      v105 = a1[9];
      v42 = (unsigned __int16 *)(*(_QWORD *)(a1[8] + 8LL) - 40LL * (unsigned int)~*(_DWORD *)(*(_QWORD *)v21 + 24LL));
      v107 = &v42[20 * *v42 + 20 + *((unsigned __int8 *)v42 + 8) + (unsigned __int64)*((unsigned int *)v42 + 3)];
      v28 = *((unsigned __int8 *)v42 + 9);
      v43 = *(_QWORD *)(*(_QWORD *)v21 + 40LL);
      v44 = v43 + 40LL * *(unsigned int *)(*(_QWORD *)v21 + 64LL);
      if ( v43 == v44 )
      {
LABEL_146:
        if ( !v28 )
          goto LABEL_129;
        v106 = 0;
      }
      else
      {
        while ( *(_DWORD *)(*(_QWORD *)v43 + 24LL) != 10 )
        {
          v43 += 40;
          if ( v44 == v43 )
            goto LABEL_146;
        }
        v106 = *(_QWORD *)(*(_QWORD *)v43 + 96LL);
        if ( !v28 && !v106 )
        {
LABEL_129:
          a6 = *(unsigned __int8 *)(v37 + 248);
          if ( (a6 & 8) == 0 )
            goto LABEL_137;
          v28 = *(_QWORD *)(a1[8] + 8LL) - 40LL * (unsigned int)~*(_DWORD *)(*(_QWORD *)v37 + 24LL);
          v73 = *(unsigned __int8 *)(v28 + 4);
          v74 = *(unsigned __int16 *)(v28 + 2);
          if ( v73 == v74 )
            goto LABEL_137;
          v41 = v99;
          a2 = 0;
          while ( 1 )
          {
            if ( v74 > v73
              && (*(_BYTE *)(v28
                           + 6LL * *(unsigned __int16 *)(v28 + 16)
                           + 8 * (5LL * *(unsigned __int16 *)v28 + 5)
                           + 6LL * v73
                           + 4)
                & 1) != 0 )
            {
              v75 = *(int *)(*(__int64 *)((char *)a2->m128i_i64 + *(_QWORD *)(*(_QWORD *)v37 + 40LL)) + 36);
              if ( (_DWORD)v75 != -1 )
              {
                v33 = *(_QWORD *)a1[6] + (v75 << 8);
                if ( *(_QWORD *)(v99 + 8) == v33 )
                  break;
              }
            }
            ++v73;
            a2 = (__m128i *)((char *)a2 + 40);
            if ( v74 == v73 )
              goto LABEL_137;
          }
          if ( v101 && !(unsigned __int8)sub_3351570(v37)
            || (*(_BYTE *)(v21 + 248) & 0x10) == 0 && (a6 &= 0x10u, (_DWORD)a6) )
          {
LABEL_137:
            a2 = (__m128i *)v37;
            if ( !(unsigned __int8)sub_2F90B20(a1[11] + 792LL, v37, v21, v28, v41, a6) )
            {
              v120.m128i_i64[1] = 3;
              v113 = a1[11];
              v120.m128i_i64[0] = v37 | 6;
              sub_2F8FA50(v113 + 792, v21, v37, v28, v76, a6);
              a2 = &v120;
              sub_2F8F1B0(v21, (__int64)&v120, 1u, v77, v78, v79);
            }
          }
          goto LABEL_50;
        }
      }
      v45 = *(_QWORD **)(v21 + 120);
      v96 = &v45[2 * *(unsigned int *)(v21 + 128)];
      if ( v45 == v96 )
        goto LABEL_129;
      v100 = *(_QWORD **)(v21 + 120);
      v98 = v21;
      v46 = &v107[v28];
      v97 = v34;
      v47 = v41 + 792;
      while ( 1 )
      {
        v48 = *v100 & 0xFFFFFFFFFFFFFFF8LL;
        v41 = *(_QWORD *)(v48 + 40);
        v49 = 16LL * *(unsigned int *)(v48 + 48);
        v50 = v41;
        v112 = v41 + v49;
        if ( v41 != v41 + v49 )
          break;
LABEL_127:
        v100 += 2;
        if ( v96 == v100 )
        {
          v21 = v98;
          v34 = v97;
          goto LABEL_129;
        }
      }
      while ( 1 )
      {
        if ( (*(_QWORD *)v50 & 6) == 0 )
        {
          v51 = *(_DWORD *)(v50 + 8);
          if ( v51 )
          {
            if ( v106 )
            {
              v28 = *(unsigned int *)(v106 + 4LL * (v51 >> 5));
              if ( !_bittest((const int *)&v28, v51) )
              {
                a2 = (__m128i *)v37;
                if ( (unsigned __int8)sub_2F90B20(v47, v37, *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL, v28, v41, a6) )
                  goto LABEL_88;
              }
            }
            v52 = v107;
            if ( v46 != v107 )
              break;
          }
        }
LABEL_126:
        v50 += 16;
        if ( v112 == v50 )
          goto LABEL_127;
      }
      while ( 1 )
      {
        v53 = *(_DWORD *)(v50 + 8);
        v54 = *v52;
        if ( v53 == v54 || v54 - 1 <= 0x3FFFFFFE && v53 - 1 <= 0x3FFFFFFE && (unsigned __int8)sub_E92070(v105, v54, v53) )
        {
          a2 = (__m128i *)v37;
          if ( (unsigned __int8)sub_2F90B20(v47, v37, *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL, v28, v41, a6) )
            break;
        }
        if ( v46 == ++v52 )
          goto LABEL_126;
      }
LABEL_88:
      v21 = v98;
      v34 = v97 + 2;
      if ( v114 == v97 + 2 )
      {
LABEL_89:
        a2 = (__m128i *)a1[6];
        goto LABEL_90;
      }
    }
  }
  v110 = a2->m128i_i64[1];
LABEL_3:
  if ( !*((_BYTE *)v6 + 44) && !*((_BYTE *)v6 + 45) )
  {
    v55 = a2->m128i_i64[0];
    v56 = v110;
    if ( a2->m128i_i64[0] != v110 )
    {
      v57 = v6;
      do
      {
        v58 = *(_DWORD *)(v55 + 212);
        if ( !v58 && *(_DWORD *)(v55 + 208) == 1 )
        {
          v59 = *(_QWORD *)v55;
          if ( !*(_QWORD *)v55
            || *(_DWORD *)(v59 + 24) != 49
            || (a6 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v59 + 40) + 40LL) + 96LL), (int)a6 >= 0) )
          {
            v60 = *(_QWORD **)(v55 + 40);
            v61 = &v60[2 * *(unsigned int *)(v55 + 48)];
            if ( v60 == v61 )
LABEL_162:
              BUG();
            v62 = *(_QWORD **)(v55 + 40);
            while ( 1 )
            {
              if ( (*v62 & 6) != 0 )
              {
                v63 = (__int64 *)(*v62 & 0xFFFFFFFFFFFFFFF8LL);
                if ( v63 )
                {
                  v64 = *v63;
                  if ( v64 )
                  {
                    v65 = *(_DWORD *)(v64 + 24);
                    if ( v65 < 0 && ~v65 == *(_DWORD *)(v57[8] + 64LL) )
                      break;
                  }
                }
              }
              v62 += 2;
              if ( v61 == v62 )
              {
                while ( (*v60 & 6) != 0 )
                {
                  v60 += 2;
                  if ( v61 == v60 )
                    goto LABEL_162;
                }
                v66 = *v60 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v66 + 248) & 0x40) == 0
                  && *(_DWORD *)(v66 + 212) != 1
                  && (!v59
                   || *(_DWORD *)(v59 + 24) != 50
                   || *(int *)(*(_QWORD *)(*(_QWORD *)(v59 + 40) + 40LL) + 96LL) >= 0) )
                {
                  v67 = *(_QWORD **)(v66 + 120);
                  v68 = v67;
                  v115 = &v67[2 * *(unsigned int *)(v66 + 128)];
                  if ( v67 == v115 )
                  {
LABEL_156:
                    v80 = 0;
                    if ( *(_DWORD *)(v66 + 128) )
                    {
                      do
                      {
                        v94 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v66 + 120) + 16 * v80));
                        v120 = v94;
                        if ( v55 == (v94.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) )
                        {
                          ++v58;
                        }
                        else
                        {
                          v119 = v56;
                          v120.m128i_i64[0] = v66 | v94.m128i_i8[0] & 7;
                          nullsub_1666();
                          sub_2F8F420(v94.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, &v120);
                          sub_2F8FA50(v57[11] + 792LL, v55, v120.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, v81, v82, v83);
                          sub_2F8F1B0(v55, (__int64)&v120, 1u, v84, v85, v86);
                          v87 = v57[11] + 792LL;
                          v120.m128i_i64[0] = v55 | v120.m128i_i8[0] & 7;
                          sub_2F8FA50(
                            v87,
                            v94.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL,
                            v120.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL,
                            v88,
                            v89,
                            v90);
                          sub_2F8F1B0(v94.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, (__int64)&v120, 1u, v91, v92, v93);
                          v56 = v119;
                        }
                        v80 = v58;
                      }
                      while ( *(_DWORD *)(v66 + 128) != v58 );
                    }
                  }
                  else
                  {
                    while ( 1 )
                    {
                      v70 = *v68 & 0xFFFFFFFFFFFFFFF8LL;
                      if ( v55 != v70 )
                      {
                        v71 = *(unsigned int *)(v70 + 212);
                        if ( !(_DWORD)v71 )
                          break;
                        if ( *(char *)(v55 + 248) < 0 && (*(_BYTE *)(v70 + 248) & 0x40) != 0 )
                        {
                          v111 = v56;
                          v118 = *v68 & 0xFFFFFFFFFFFFFFF8LL;
                          v72 = sub_3353030(*(_QWORD *)v70, *(_QWORD *)v55, v57[8], v57[9], v56);
                          v70 = v118;
                          v56 = v111;
                          if ( v72 )
                            break;
                        }
                        v117 = v56;
                        v69 = sub_2F90B20(v57[11] + 792LL, v55, v70, v71, v56, a6);
                        v56 = v117;
                        if ( v69 )
                          break;
                      }
                      v68 += 2;
                      if ( v115 == v68 )
                        goto LABEL_156;
                    }
                  }
                }
                break;
              }
            }
          }
        }
        v55 += 256;
      }
      while ( v56 != v55 );
      a2 = (__m128i *)v57[6];
      v6 = v57;
      v110 = a2->m128i_i64[1];
    }
  }
  v120.m128i_i32[0] = 0;
  sub_1D05C60((__int64)(v6 + 12), (v110 - a2->m128i_i64[0]) >> 8, v120.m128i_i32);
  v7 = (__int64 *)v6[6];
  v8 = v7[1];
  for ( i = *v7; v8 != i; i += 256 )
  {
    while ( *(_DWORD *)(v6[12] + 4LL * *(unsigned int *)(i + 200)) )
    {
      i += 256;
      if ( v8 == i )
        goto LABEL_10;
    }
    v10 = i;
    sub_3352560(v10, v6 + 12);
  }
LABEL_10:
  LOBYTE(v11) = sub_2E322C0(*(_QWORD *)(v6[11] + 584LL), *(_QWORD *)(v6[11] + 584LL));
  v12 = v11;
  if ( (_BYTE)v11 )
  {
    LOBYTE(v11) = (_BYTE)v104;
    v13 = *v104;
    v14 = v104[1];
    if ( *v104 != v14 )
    {
      v15 = qword_5038C68;
      do
      {
        while ( 1 )
        {
          if ( !v15 )
          {
            v16 = *(__int64 **)(v13 + 40);
            v17 = &v16[2 * *(unsigned int *)(v13 + 48)];
            if ( v16 != v17 )
            {
              v18 = *(__int64 **)(v13 + 40);
              v19 = 0;
              do
              {
                while ( 1 )
                {
                  v11 = *v18;
                  if ( (*v18 & 6) == 0 )
                    break;
                  v18 += 2;
                  if ( v17 == v18 )
                    goto LABEL_22;
                }
                v11 = *(_QWORD *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
                if ( !v11 )
                  goto LABEL_16;
                if ( *(_DWORD *)(v11 + 24) != 50 )
                  goto LABEL_16;
                LODWORD(v11) = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 40) + 40LL) + 96LL);
                if ( (int)v11 >= 0 )
                  goto LABEL_16;
                v18 += 2;
                v19 = v12;
              }
              while ( v17 != v18 );
LABEL_22:
              if ( v19 )
              {
                LOBYTE(v11) = sub_3351570(v13);
                if ( (_BYTE)v11 )
                  break;
              }
            }
          }
LABEL_16:
          v13 += 256;
          if ( v14 == v13 )
            return v11;
        }
        *(_BYTE *)(v13 + 248) |= 1u;
        do
        {
          v11 = *v16;
          if ( (*v16 & 6) == 0 )
          {
            v11 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_BYTE *)(v11 + 248) |= 1u;
          }
          v16 += 2;
        }
        while ( v20 != v16 );
        v13 += 256;
      }
      while ( v14 != v13 );
    }
  }
  return v11;
}
