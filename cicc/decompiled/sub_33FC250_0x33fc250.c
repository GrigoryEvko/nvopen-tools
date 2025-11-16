// Function: sub_33FC250
// Address: 0x33fc250
//
unsigned __int8 *__fastcall sub_33FC250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char *a4,
        __int64 a5,
        _QWORD *a6,
        __m128i a7)
{
  char *v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *v11; // rax
  char *v12; // rcx
  __int64 v13; // r9
  unsigned int *v14; // rdx
  unsigned int *v15; // rcx
  unsigned int *v16; // rbx
  unsigned int *v17; // rsi
  int v18; // r11d
  int v19; // r15d
  unsigned __int16 v20; // r13
  unsigned int **v21; // r8
  unsigned __int8 *v22; // r12
  __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int16 v25; // di
  __int64 v26; // rax
  __int64 v27; // r14
  unsigned __int8 *result; // rax
  __int64 v29; // rdi
  unsigned int *v30; // rbx
  unsigned int *v31; // r15
  unsigned int *v32; // r12
  _QWORD *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rax
  unsigned int *v39; // r13
  __int64 v40; // rdx
  __int64 v41; // rax
  __int16 v42; // cx
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rax
  const __m128i *v46; // r12
  __int64 v47; // rdx
  const __m128i *v48; // r14
  __int64 v49; // r8
  int v50; // r15d
  __m128 *v51; // rax
  unsigned int *v52; // rbx
  unsigned int *v53; // r12
  unsigned __int16 i; // r14
  __int64 v55; // rax
  unsigned __int16 v56; // r13
  __int64 v57; // r15
  unsigned __int64 v58; // r8
  __int64 v59; // rdx
  char v60; // r9
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  __int64 v66; // r11
  _QWORD *v67; // rax
  _QWORD *v68; // r14
  unsigned int v69; // edx
  unsigned int v70; // r15d
  __int64 v71; // rax
  __int64 v72; // r12
  unsigned __int64 v73; // r8
  int v74; // edx
  unsigned int *v75; // rax
  unsigned __int16 v76; // bx
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // rdx
  char v81; // al
  __int64 v82; // r9
  __int64 v83; // rcx
  __int64 *v84; // r14
  __int64 *v85; // rbx
  unsigned int v86; // r12d
  __int64 v87; // rdi
  __int64 (*v88)(); // r9
  __int64 v89; // rax
  unsigned __int8 *v90; // rsi
  int v91; // edx
  int v92; // eax
  _QWORD *v93; // rax
  int v94; // edx
  int v95; // ecx
  _QWORD *v96; // r10
  int v97; // edx
  __int64 v98; // rdx
  __int128 v99; // [rsp-10h] [rbp-220h]
  __int16 v100; // [rsp+2h] [rbp-20Eh]
  unsigned int *v101; // [rsp+10h] [rbp-200h]
  unsigned int *v102; // [rsp+18h] [rbp-1F8h]
  int v103; // [rsp+20h] [rbp-1F0h]
  int v104; // [rsp+20h] [rbp-1F0h]
  unsigned int **v105; // [rsp+28h] [rbp-1E8h]
  unsigned __int64 v106; // [rsp+28h] [rbp-1E8h]
  _QWORD *v107; // [rsp+28h] [rbp-1E8h]
  unsigned int v108; // [rsp+28h] [rbp-1E8h]
  int v109; // [rsp+30h] [rbp-1E0h]
  unsigned __int16 v110; // [rsp+30h] [rbp-1E0h]
  char v111; // [rsp+30h] [rbp-1E0h]
  int v112; // [rsp+38h] [rbp-1D8h]
  unsigned __int8 *v115; // [rsp+48h] [rbp-1C8h]
  __int64 v116; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v117; // [rsp+78h] [rbp-198h]
  __int64 v118; // [rsp+88h] [rbp-188h]
  __int64 v119; // [rsp+90h] [rbp-180h] BYREF
  __int64 v120; // [rsp+98h] [rbp-178h]
  __int64 v121; // [rsp+A0h] [rbp-170h]
  __int64 v122; // [rsp+A8h] [rbp-168h]
  __int64 v123; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v124; // [rsp+B8h] [rbp-158h]
  unsigned int *v125; // [rsp+C0h] [rbp-150h] BYREF
  __int64 v126; // [rsp+C8h] [rbp-148h]
  unsigned int *v127; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v128; // [rsp+D8h] [rbp-138h]
  _BYTE v129[304]; // [rsp+E0h] [rbp-130h] BYREF

  v116 = a2;
  v117 = a3;
  if ( a5 == 1 )
    return *(unsigned __int8 **)a4;
  v8 = &a4[16 * a5];
  v9 = (16 * a5) >> 6;
  v10 = (16 * a5) >> 4;
  if ( v9 <= 0 )
  {
    v11 = a4;
LABEL_25:
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
          goto LABEL_28;
        goto LABEL_103;
      }
      if ( *(_DWORD *)(*(_QWORD *)v11 + 24LL) != 51 )
      {
LABEL_9:
        if ( v8 != v11 )
          goto LABEL_10;
        goto LABEL_28;
      }
      v11 += 16;
    }
    if ( *(_DWORD *)(*(_QWORD *)v11 + 24LL) == 51 )
    {
      v11 += 16;
LABEL_103:
      if ( *(_DWORD *)(*(_QWORD *)v11 + 24LL) == 51 )
        goto LABEL_28;
      goto LABEL_9;
    }
    goto LABEL_9;
  }
  v11 = a4;
  v12 = &a4[64 * v9];
  while ( 1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)v11 + 24LL) != 51 )
      goto LABEL_9;
    if ( *(_DWORD *)(*((_QWORD *)v11 + 2) + 24LL) != 51 )
    {
      if ( v8 != v11 + 16 )
        goto LABEL_10;
      goto LABEL_28;
    }
    if ( *(_DWORD *)(*((_QWORD *)v11 + 4) + 24LL) != 51 )
    {
      if ( v8 != v11 + 32 )
        goto LABEL_10;
LABEL_28:
      v127 = 0;
      LODWORD(v128) = 0;
      v33 = sub_33F17F0(a6, 51, (__int64)&v127, v116, v117);
      if ( v127 )
        sub_B91220((__int64)&v127, (__int64)v127);
      return (unsigned __int8 *)v33;
    }
    if ( *(_DWORD *)(*((_QWORD *)v11 + 6) + 24LL) != 51 )
      break;
    v11 += 64;
    if ( v12 == v11 )
    {
      v10 = (v8 - v11) >> 4;
      goto LABEL_25;
    }
  }
  if ( v8 == v11 + 48 )
    goto LABEL_28;
LABEL_10:
  v109 = a5;
  if ( !(_DWORD)a5 )
    return 0;
  v13 = 0;
  v14 = (unsigned int *)&a4[16 * a5];
  v15 = (unsigned int *)a4;
  v16 = (unsigned int *)a4;
  v17 = (unsigned int *)a4;
  v18 = 0;
  v19 = 0;
  v20 = v116;
  v21 = &v127;
  v22 = 0;
  while ( 2 )
  {
    v23 = *(_QWORD *)v16;
    v24 = *(_QWORD *)(*(_QWORD *)v16 + 48LL) + 16LL * v16[2];
    v25 = *(_WORD *)v24;
    v26 = *(_QWORD *)(v24 + 8);
    LOWORD(v127) = v25;
    v128 = v26;
    if ( v25 )
    {
      v112 = word_4456340[v25 - 1];
    }
    else
    {
      v101 = v15;
      v102 = v14;
      v103 = v18;
      v105 = v21;
      v34 = sub_3007240((__int64)v21);
      v15 = v101;
      v118 = v34;
      v14 = v102;
      v112 = v34;
      v18 = v103;
      v21 = v105;
    }
    if ( *(_DWORD *)(v23 + 24) != 161
      || (v27 = *(_QWORD *)(v23 + 40),
          result = *(unsigned __int8 **)v27,
          v29 = *(unsigned int *)(v27 + 8),
          v13 = *(_QWORD *)(*(_QWORD *)v27 + 48LL) + 16 * v29,
          *(_WORD *)v13 != (_WORD)v116) )
    {
LABEL_21:
      v30 = v14;
      v31 = v15;
      v32 = v17;
      if ( !(_WORD)v116 )
        goto LABEL_35;
      if ( (unsigned __int16)(v116 - 176) <= 0x34u )
        return 0;
      if ( (unsigned __int16)(v116 - 17) > 0xD3u )
      {
LABEL_37:
        v38 = v117;
      }
      else
      {
        v20 = word_4456580[(unsigned __int16)v116 - 1];
        v38 = 0;
      }
LABEL_38:
      v120 = v38;
      v127 = (unsigned int *)v129;
      LOWORD(v119) = v20;
      v128 = 0x1000000000LL;
      if ( v30 == v32 )
        goto LABEL_109;
      v110 = v20;
      v39 = v31;
      while ( 1 )
      {
        v40 = *(_QWORD *)v39;
        v41 = *(_QWORD *)(*(_QWORD *)v39 + 48LL) + 16LL * v39[2];
        v42 = *(_WORD *)v41;
        v43 = *(_QWORD *)(v41 + 8);
        LOWORD(v123) = v42;
        v124 = v43;
        v44 = *(_DWORD *)(v40 + 24);
        if ( v44 == 51 )
          break;
        if ( v44 != 156 )
        {
          result = 0;
          goto LABEL_124;
        }
        v45 = *(unsigned int *)(v40 + 64);
        v46 = *(const __m128i **)(v40 + 40);
        v47 = (unsigned int)v128;
        v48 = (const __m128i *)((char *)v46 + 40 * v45);
        v49 = (40 * v45) >> 3;
        v50 = -858993459 * v49;
        if ( 0xCCCCCCCCCCCCCCCDLL * v49 + (unsigned int)v128 > HIDWORD(v128) )
        {
          v17 = (unsigned int *)v129;
          sub_C8D5F0((__int64)&v127, v129, 0xCCCCCCCCCCCCCCCDLL * v49 + (unsigned int)v128, 0x10u, v49, v13);
          v47 = (unsigned int)v128;
        }
        v51 = (__m128 *)&v127[4 * v47];
        if ( v46 != v48 )
        {
          do
          {
            if ( v51 )
            {
              a7 = _mm_loadu_si128(v46);
              *v51 = (__m128)a7;
            }
            v46 = (const __m128i *)((char *)v46 + 40);
            ++v51;
          }
          while ( v48 != v46 );
          LODWORD(v47) = v128;
        }
        LODWORD(v128) = v50 + v47;
LABEL_50:
        v39 += 4;
        if ( v30 == v39 )
        {
          v52 = v127;
          v53 = &v127[4 * (unsigned int)v128];
          if ( v53 != v127 )
          {
            for ( i = v110; ; i = v119 )
            {
              v55 = *(_QWORD *)(*(_QWORD *)v52 + 48LL) + 16LL * v52[2];
              v56 = *(_WORD *)v55;
              v57 = *(_QWORD *)(v55 + 8);
              if ( *(_WORD *)v55 == i )
              {
                if ( i || v57 == v120 )
                  goto LABEL_54;
                v126 = *(_QWORD *)(v55 + 8);
                LOWORD(v125) = 0;
              }
              else
              {
                LOWORD(v125) = *(_WORD *)v55;
                v126 = v57;
                if ( v56 )
                {
                  if ( v56 == 1 || (unsigned __int16)(v56 - 504) <= 7u )
LABEL_139:
                    BUG();
                  v58 = *(_QWORD *)&byte_444C4A0[16 * v56 - 16];
                  v60 = byte_444C4A0[16 * v56 - 8];
                  goto LABEL_59;
                }
              }
              v123 = sub_3007260((__int64)&v125);
              v58 = v123;
              v124 = v59;
              v60 = v59;
LABEL_59:
              if ( i )
              {
                if ( i == 1 || (unsigned __int16)(i - 504) <= 7u )
                  goto LABEL_139;
                v65 = *(_QWORD *)&byte_444C4A0[16 * i - 16];
                LOBYTE(v64) = byte_444C4A0[16 * i - 8];
              }
              else
              {
                v106 = v58;
                v111 = v60;
                v61 = sub_3007260((__int64)&v119);
                v58 = v106;
                v60 = v111;
                v62 = v61;
                v64 = v63;
                v121 = v62;
                v65 = v62;
                v122 = v64;
              }
              if ( (!(_BYTE)v64 || v60) && v58 > v65 )
              {
                LOWORD(v119) = v56;
                v120 = v57;
              }
LABEL_54:
              v52 += 4;
              if ( v53 == v52 )
                break;
            }
          }
LABEL_109:
          v76 = v116;
          if ( (_WORD)v116 )
          {
            if ( (unsigned __int16)(v116 - 17) > 0xD3u )
            {
LABEL_111:
              v80 = v117;
            }
            else
            {
              v76 = word_4456580[(unsigned __int16)v116 - 1];
              v80 = 0;
            }
          }
          else
          {
            if ( !sub_30070B0((__int64)&v116) )
              goto LABEL_111;
            v76 = sub_3009970((__int64)&v116, (__int64)v17, v77, v78, v79);
          }
          v81 = sub_3280A00((__int64)&v119, v76, v80);
          v83 = (unsigned int)v128;
          v84 = (__int64 *)v127;
          if ( v81 )
          {
            v85 = (__int64 *)&v127[4 * (unsigned int)v128];
            if ( v85 != (__int64 *)v127 )
            {
              HIWORD(v86) = v100;
              do
              {
                if ( *(_DWORD *)(*v84 + 24) == 51 )
                {
                  v125 = 0;
                  LODWORD(v126) = 0;
                  v93 = sub_33F17F0(a6, 51, (__int64)&v125, v119, v120);
                  v95 = v94;
                  v96 = v93;
                  if ( v125 )
                  {
                    v104 = v94;
                    v107 = v93;
                    sub_B91220((__int64)&v125, (__int64)v125);
                    v95 = v104;
                    v96 = v107;
                  }
                  *v84 = (__int64)v96;
                  *((_DWORD *)v84 + 2) = v95;
                }
                else
                {
                  v87 = a6[2];
                  v88 = *(__int64 (**)())(*(_QWORD *)v87 + 1432LL);
                  v89 = *(_QWORD *)(*v84 + 48) + 16LL * *((unsigned int *)v84 + 2);
                  if ( v88 == sub_2FE34A0
                    || (LOWORD(v86) = *(_WORD *)v89,
                        !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, __int64))v88)(
                           v87,
                           v86,
                           *(_QWORD *)(v89 + 8),
                           (unsigned int)v119,
                           v120)) )
                  {
                    v90 = sub_33FB160((__int64)a6, *v84, v84[1], a1, (unsigned int)v119, v120, a7);
                    v92 = v91;
                  }
                  else
                  {
                    v90 = sub_33FB310((__int64)a6, *v84, v84[1], a1, (unsigned int)v119, v120, a7);
                    v92 = v97;
                  }
                  *v84 = (__int64)v90;
                  *((_DWORD *)v84 + 2) = v92;
                }
                v84 += 2;
              }
              while ( v85 != v84 );
              v84 = (__int64 *)v127;
              v83 = (unsigned int)v128;
            }
          }
          *((_QWORD *)&v99 + 1) = v83;
          *(_QWORD *)&v99 = v84;
          result = sub_33FC220(a6, 156, a1, v116, v117, v82, v99);
LABEL_124:
          if ( v127 != (unsigned int *)v129 )
          {
            v115 = result;
            _libc_free((unsigned __int64)v127);
            return v115;
          }
          return result;
        }
      }
      v125 = 0;
      LODWORD(v126) = 0;
      v67 = sub_33F17F0(a6, 51, (__int64)&v125, v119, v120);
      v17 = v125;
      v68 = v67;
      v70 = v69;
      if ( v125 )
        sub_B91220((__int64)&v125, (__int64)v125);
      if ( (_WORD)v123 )
      {
        if ( (unsigned __int16)(v123 - 176) > 0x34u )
        {
LABEL_76:
          v13 = word_4456340[(unsigned __int16)v123 - 1];
LABEL_77:
          v71 = (unsigned int)v128;
          v72 = (unsigned int)v13;
          v73 = (unsigned int)v13 + (unsigned __int64)(unsigned int)v128;
          v74 = v128;
          if ( v73 > HIDWORD(v128) )
          {
            v17 = (unsigned int *)v129;
            v108 = v13;
            sub_C8D5F0((__int64)&v127, v129, (unsigned int)v13 + (unsigned __int64)(unsigned int)v128, 0x10u, v73, v13);
            v71 = (unsigned int)v128;
            v13 = v108;
            v74 = v128;
          }
          v75 = &v127[4 * v71];
          if ( v72 )
          {
            do
            {
              if ( v75 )
              {
                *(_QWORD *)v75 = v68;
                v75[2] = v70;
              }
              v75 += 4;
              --v72;
            }
            while ( v72 );
            v74 = v128;
          }
          LODWORD(v128) = v13 + v74;
          goto LABEL_50;
        }
      }
      else if ( !sub_3007100((__int64)&v123) )
      {
        goto LABEL_86;
      }
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      if ( (_WORD)v123 )
      {
        if ( (unsigned __int16)(v123 - 176) <= 0x34u )
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        goto LABEL_76;
      }
LABEL_86:
      v13 = (unsigned int)sub_3007130((__int64)&v123, (__int64)v17);
      goto LABEL_77;
    }
    if ( *(_QWORD *)(v13 + 8) == v117 || (_WORD)v116 )
    {
      if ( v22 && (result != v22 || (_DWORD)v29 != v18) )
        goto LABEL_21;
      v66 = *(_QWORD *)(*(_QWORD *)(v27 + 40) + 96LL);
      v13 = *(_QWORD *)(v66 + 24);
      if ( *(_DWORD *)(v66 + 32) > 0x40u )
        v13 = *(_QWORD *)v13;
      if ( v19 * v112 != v13 )
        goto LABEL_21;
      ++v19;
      v16 += 4;
      if ( v109 == v19 )
        return result;
      v18 = *(_DWORD *)(v27 + 8);
      v22 = *(unsigned __int8 **)v27;
      continue;
    }
    break;
  }
  v30 = v14;
  v31 = v15;
  v32 = v17;
LABEL_35:
  if ( !sub_3007100((__int64)&v116) )
  {
    if ( !sub_30070B0((__int64)&v116) )
      goto LABEL_37;
    v20 = sub_3009970((__int64)&v116, (__int64)v17, v35, v36, v37);
    v38 = v98;
    goto LABEL_38;
  }
  return 0;
}
