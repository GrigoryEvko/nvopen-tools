// Function: sub_1D05DD0
// Address: 0x1d05dd0
//
__int64 __fastcall sub_1D05DD0(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r14
  __int64 *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 result; // rax
  char v8; // r10
  __int64 v9; // rdi
  __int64 v10; // r11
  char v11; // bl
  __int64 *v12; // rsi
  __int64 *v13; // r9
  __int64 *v14; // rdx
  char v15; // cl
  __int64 *v16; // r9
  __int64 v17; // r13
  __int64 v18; // rax
  int v19; // edx
  unsigned int *v20; // rax
  __int16 v21; // r9
  __int64 v22; // rax
  unsigned int v23; // ebx
  unsigned int v24; // eax
  __int64 v25; // r15
  int v26; // eax
  __int64 v27; // rax
  bool v28; // zf
  __int64 v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r13
  _QWORD *v32; // r15
  unsigned int v33; // r14d
  unsigned int v34; // ebx
  unsigned __int64 v35; // r12
  __int64 v36; // rdi
  __int16 v37; // ax
  __int64 v38; // rax
  __int64 v39; // rcx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 v44; // r15
  __int64 v45; // rbx
  __int64 v46; // r15
  __int64 v47; // r14
  unsigned int v48; // eax
  int v49; // ecx
  unsigned __int16 v50; // ax
  unsigned __int64 v51; // rbx
  unsigned __int64 v52; // r12
  int v53; // esi
  __int64 v54; // rcx
  __int64 v55; // r9
  unsigned int v56; // edi
  _WORD *v57; // r10
  unsigned __int16 v58; // ax
  _WORD *v59; // rdi
  unsigned int v60; // edx
  _WORD *v61; // r10
  unsigned int v62; // ecx
  int v63; // edx
  unsigned __int16 *v64; // r9
  unsigned int i; // esi
  bool v66; // cf
  char v67; // r10
  unsigned __int64 v68; // r9
  unsigned int v69; // edx
  unsigned int v70; // edi
  __int64 v71; // rsi
  int v72; // eax
  __int64 v73; // r13
  _QWORD *v74; // r12
  unsigned int v75; // ebx
  __int64 v76; // rcx
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  unsigned __int64 v79; // r15
  _QWORD *v80; // r11
  __int64 v81; // rdx
  _QWORD *v82; // r14
  unsigned __int64 v83; // r10
  char v84; // al
  int v85; // esi
  __int64 v86; // rax
  __int64 v87; // rdi
  __m128i v88; // xmm0
  unsigned __int64 v89; // r14
  _QWORD *v90; // [rsp+8h] [rbp-C8h]
  _QWORD *v91; // [rsp+18h] [rbp-B8h]
  __int64 v92; // [rsp+20h] [rbp-B0h]
  __int64 v93; // [rsp+28h] [rbp-A8h]
  __int64 v94; // [rsp+30h] [rbp-A0h]
  _QWORD *v95; // [rsp+38h] [rbp-98h]
  char v96; // [rsp+43h] [rbp-8Dh]
  unsigned int v97; // [rsp+44h] [rbp-8Ch]
  __int64 *v99; // [rsp+50h] [rbp-80h]
  __int64 v100; // [rsp+58h] [rbp-78h]
  __int64 v101; // [rsp+60h] [rbp-70h]
  unsigned __int64 v102; // [rsp+68h] [rbp-68h]
  __int64 v103; // [rsp+70h] [rbp-60h]
  __int64 v104; // [rsp+70h] [rbp-60h]
  unsigned __int64 v105; // [rsp+70h] [rbp-60h]
  __int64 v106; // [rsp+78h] [rbp-58h]
  unsigned int v107; // [rsp+80h] [rbp-50h]
  _QWORD *v108; // [rsp+88h] [rbp-48h]
  _QWORD *v109; // [rsp+88h] [rbp-48h]
  __m128i v110; // [rsp+90h] [rbp-40h] BYREF

  v2 = a1;
  a1[6] = a2;
  v99 = a2;
  if ( !byte_4FC0CA0 )
  {
    v106 = a2[1];
    if ( *a2 == v106 )
      goto LABEL_3;
    v17 = *a2;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v17 + 228) & 8) != 0 )
      {
        v18 = *(_QWORD *)v17;
        if ( *(_QWORD *)v17 )
        {
          if ( *(__int16 *)(v18 + 24) < 0 )
          {
            v19 = *(_DWORD *)(v18 + 56);
            if ( !v19
              || (v20 = (unsigned int *)(*(_QWORD *)(v18 + 32) + 40LL * (unsigned int)(v19 - 1)),
                  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v20 + 40LL) + 16LL * v20[2]) != 111) )
            {
              v96 = sub_1D00EC0(v17);
              v22 = *(_QWORD *)(a1[8] + 8LL) + ((__int64)~v21 << 6);
              v23 = *(unsigned __int8 *)(v22 + 4);
              v100 = v22;
              v24 = *(unsigned __int16 *)(v22 + 2);
              v107 = v23;
              v97 = v24;
              if ( v23 != v24 )
                break;
            }
          }
        }
      }
LABEL_57:
      v17 += 272;
      if ( v106 == v17 )
      {
        v2 = a1;
        v106 = a2[1];
        goto LABEL_3;
      }
    }
    v101 = 0;
    v25 = v17;
    while ( 1 )
    {
      if ( v24 > v107 && (*(_BYTE *)(*(_QWORD *)(v100 + 40) + 8LL * v107 + 4) & 1) != 0 )
      {
        v26 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v25 + 32LL) + v101) + 28LL);
        if ( v26 != -1 )
        {
          v27 = 272LL * v26;
          v28 = *a2 + v27 == 0;
          v29 = *a2 + v27;
          v92 = v29;
          if ( !v28 )
          {
            v30 = *(_QWORD **)(v29 + 112);
            v108 = &v30[2 * *(unsigned int *)(v29 + 120)];
            if ( v30 != v108 )
              break;
          }
        }
      }
LABEL_135:
      ++v107;
      v101 += 40;
      if ( v97 == v107 )
      {
        v17 = v25;
        goto LABEL_57;
      }
      v24 = *(unsigned __int16 *)(v100 + 2);
    }
    v31 = v25;
    v32 = *(_QWORD **)(v29 + 112);
    while ( 1 )
    {
      if ( (*v32 & 6) == 0 )
      {
        v35 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v35 != v31 )
          break;
      }
LABEL_51:
      v32 += 2;
      if ( v108 == v32 )
      {
        v25 = v31;
        a2 = (__int64 *)a1[6];
        goto LABEL_135;
      }
    }
    if ( (*(_BYTE *)(v35 + 236) & 2) == 0 )
      sub_1F01F70(v35);
    v33 = *(_DWORD *)(v35 + 244);
    if ( (*(_BYTE *)(v31 + 236) & 2) != 0 )
    {
      v34 = *(_DWORD *)(v31 + 244);
      if ( v33 >= v34 )
        goto LABEL_60;
    }
    else
    {
      sub_1F01F70(v31);
      v34 = *(_DWORD *)(v31 + 244);
      if ( v33 >= v34 )
      {
LABEL_60:
        while ( *(_DWORD *)(v35 + 120) == 1 )
        {
          v36 = *(_QWORD *)v35;
          v37 = *(_WORD *)(*(_QWORD *)v35 + 24LL);
          if ( v37 != -12 )
            goto LABEL_62;
          v35 = **(_QWORD **)(v35 + 112) & 0xFFFFFFFFFFFFFFF8LL;
        }
        v36 = *(_QWORD *)v35;
        if ( *(_QWORD *)v35 )
        {
          v37 = *(_WORD *)(v36 + 24);
LABEL_62:
          if ( v37 < 0
            && ((*(_BYTE *)(v35 + 228) & 0x40) == 0
             || *(char *)(v31 + 228) >= 0
             || !(unsigned __int8)sub_1D02A20(v36, *(_QWORD *)v31, a1[8], a1[9]))
            && (unsigned int)(-*(__int16 *)(*(_QWORD *)v35 + 24LL) - 8) > 1
            && ~*(__int16 *)(*(_QWORD *)v35 + 24LL) != 10 )
          {
            v102 = *(_QWORD *)(*(_QWORD *)(a1[8] + 8LL)
                             + ((unsigned __int64)(unsigned int)~*(__int16 *)(*(_QWORD *)v31 + 24LL) << 6)
                             + 32);
            v38 = *(_QWORD *)(*(_QWORD *)v31 + 32LL);
            v39 = v38 + 40LL * *(unsigned int *)(*(_QWORD *)v31 + 56LL);
            if ( v38 == v39 )
            {
LABEL_146:
              v103 = 0;
              v40 = *(_QWORD *)(*(_QWORD *)(a1[8] + 8LL)
                              + ((unsigned __int64)(unsigned int)~*(__int16 *)(*(_QWORD *)v31 + 24LL) << 6)
                              + 32);
            }
            else
            {
              while ( *(_WORD *)(*(_QWORD *)v38 + 24LL) != 9 )
              {
                v38 += 40;
                if ( v39 == v38 )
                  goto LABEL_146;
              }
              v103 = *(_QWORD *)(*(_QWORD *)v38 + 88LL);
              v40 = v103 | v102;
            }
            if ( !v40
              || (v41 = *(_QWORD *)(v31 + 112),
                  v42 = 16LL * *(unsigned int *)(v31 + 120),
                  v91 = (_QWORD *)(v41 + v42),
                  v41 == v41 + v42) )
            {
LABEL_96:
              v67 = *(_BYTE *)(v35 + 228);
              if ( (v67 & 8) == 0 )
                goto LABEL_104;
              v68 = *(_QWORD *)(a1[8] + 8LL)
                  + ((unsigned __int64)(unsigned int)~*(__int16 *)(*(_QWORD *)v35 + 24LL) << 6);
              v69 = *(unsigned __int8 *)(v68 + 4);
              v70 = *(unsigned __int16 *)(v68 + 2);
              if ( v69 == v70 )
                goto LABEL_104;
              v71 = 0;
              while ( 1 )
              {
                if ( v70 > v69 && (*(_BYTE *)(*(_QWORD *)(v68 + 40) + 8LL * v69 + 4) & 1) != 0 )
                {
                  v72 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v35 + 32LL) + v71) + 28LL);
                  if ( v72 != -1 && *(_QWORD *)(v92 + 16) == *(_QWORD *)a1[6] + 272LL * v72 )
                    break;
                }
                ++v69;
                v71 += 40;
                if ( v70 == v69 )
                  goto LABEL_104;
              }
              if ( v96 && !(unsigned __int8)sub_1D00EC0(v35) || (*(_BYTE *)(v31 + 228) & 0x10) == 0 && (v67 & 0x10) != 0 )
              {
LABEL_104:
                if ( !(unsigned __int8)sub_1F03240(a1[11] + 824LL, v35, v31) )
                {
                  v110.m128i_i64[1] = 3;
                  v104 = a1[11];
                  v110.m128i_i64[0] = v35 | 6;
                  sub_1F03360(v104 + 824, v31, v35);
                  sub_1F01A00(v31, &v110, 1);
                }
              }
            }
            else
            {
              v95 = *(_QWORD **)(v31 + 112);
              v90 = v32;
              v43 = a1[9];
              v94 = a1[11] + 824LL;
LABEL_73:
              v44 = *(_QWORD *)((*v95 & 0xFFFFFFFFFFFFFFF8LL) + 32);
              v45 = v44 + 16LL * *(unsigned int *)((*v95 & 0xFFFFFFFFFFFFFFF8LL) + 40);
              if ( v44 == v45 )
                goto LABEL_94;
              v46 = v43;
              v47 = *(_QWORD *)((*v95 & 0xFFFFFFFFFFFFFFF8LL) + 32);
              while ( 2 )
              {
                if ( (*(_QWORD *)v47 & 6) != 0 )
                  goto LABEL_75;
                v48 = *(_DWORD *)(v47 + 8);
                if ( !v48 )
                  goto LABEL_75;
                if ( !v103
                  || (v49 = *(_DWORD *)(v103 + 4LL * (v48 >> 5)), _bittest(&v49, v48))
                  || !(unsigned __int8)sub_1F03240(v94, v35, *(_QWORD *)v47 & 0xFFFFFFFFFFFFFFF8LL) )
                {
                  if ( !v102 || (v50 = *(_WORD *)v102) == 0 )
                  {
LABEL_75:
                    v47 += 16;
                    if ( v45 == v47 )
                      goto LABEL_93;
                    continue;
                  }
                  v93 = v45;
                  v51 = v35;
                  v52 = v102;
LABEL_83:
                  v53 = *(_DWORD *)(v47 + 8);
                  if ( v53 == v50 )
                  {
LABEL_90:
                    if ( (unsigned __int8)sub_1F03240(v94, v51, *(_QWORD *)v47 & 0xFFFFFFFFFFFFFFF8LL) )
                      break;
                  }
                  else if ( v53 >= 0 )
                  {
                    v54 = *(_QWORD *)(v46 + 8);
                    v55 = *(_QWORD *)(v46 + 56);
                    v56 = *(_DWORD *)(v54 + 24LL * v50 + 16);
                    v57 = (_WORD *)(v55 + 2LL * (v56 >> 4));
                    v58 = *v57 + (v56 & 0xF) * v50;
                    v59 = v57 + 1;
                    v60 = *(_DWORD *)(v54 + 24LL * (unsigned int)v53 + 16);
                    v61 = (_WORD *)(v55 + 2LL * (v60 >> 4));
                    v62 = v58;
                    v63 = v53 * (v60 & 0xF);
                    LOWORD(v63) = *v61 + v63;
                    v64 = v61 + 1;
                    for ( i = (unsigned __int16)v63; ; i = (unsigned __int16)v63 )
                    {
                      v66 = v62 < i;
                      if ( v62 == i )
                        break;
                      while ( v66 )
                      {
                        v58 += *v59;
                        if ( !*v59 )
                          goto LABEL_91;
                        v62 = v58;
                        ++v59;
                        v66 = v58 < i;
                        if ( v58 == i )
                          goto LABEL_90;
                      }
                      v85 = *v64;
                      if ( !(_WORD)v85 )
                        goto LABEL_91;
                      v63 += v85;
                      ++v64;
                    }
                    goto LABEL_90;
                  }
LABEL_91:
                  v50 = *(_WORD *)(v52 + 2);
                  v52 += 2LL;
                  if ( !v50 )
                  {
                    v35 = v51;
                    v45 = v93;
                    v47 += 16;
                    if ( v93 == v47 )
                    {
LABEL_93:
                      v43 = v46;
LABEL_94:
                      v95 += 2;
                      if ( v91 == v95 )
                      {
                        v32 = v90;
                        goto LABEL_96;
                      }
                      goto LABEL_73;
                    }
                    continue;
                  }
                  goto LABEL_83;
                }
                break;
              }
              v32 = v90;
            }
          }
        }
        goto LABEL_51;
      }
      if ( (*(_BYTE *)(v31 + 236) & 2) == 0 )
      {
        sub_1F01F70(v31);
        v34 = *(_DWORD *)(v31 + 244);
        if ( (*(_BYTE *)(v35 + 236) & 2) != 0 )
          goto LABEL_50;
        goto LABEL_140;
      }
    }
    if ( (*(_BYTE *)(v35 + 236) & 2) != 0 )
      goto LABEL_50;
LABEL_140:
    sub_1F01F70(v35);
LABEL_50:
    if ( v34 - *(_DWORD *)(v35 + 244) <= 1 )
      goto LABEL_60;
    goto LABEL_51;
  }
  v106 = a2[1];
LABEL_3:
  if ( !*((_BYTE *)v2 + 44) && !*((_BYTE *)v2 + 45) && *a2 != v106 )
  {
    v73 = *a2;
    v74 = v2;
    do
    {
      v75 = *(_DWORD *)(v73 + 204);
      if ( *(_QWORD *)(v73 + 200) == 1 )
      {
        v76 = *(_QWORD *)v73;
        if ( !*(_QWORD *)v73
          || *(_WORD *)(v76 + 24) != 46
          || *(int *)(*(_QWORD *)(*(_QWORD *)(v76 + 32) + 40LL) + 84LL) >= 0 )
        {
          v77 = *(_QWORD **)(v73 + 32);
          v78 = &v77[2 * *(unsigned int *)(v73 + 40)];
          if ( v77 == v78 )
LABEL_162:
            BUG();
          while ( (*v77 & 6) != 0 )
          {
            v77 += 2;
            if ( v78 == v77 )
              goto LABEL_162;
          }
          v79 = *v77 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v79 + 228) & 0x40) == 0
            && *(_DWORD *)(v79 + 204) != 1
            && (!v76 || *(_WORD *)(v76 + 24) != 47 || *(int *)(*(_QWORD *)(*(_QWORD *)(v76 + 32) + 40LL) + 84LL) >= 0) )
          {
            v80 = *(_QWORD **)(v79 + 112);
            v81 = *(unsigned int *)(v79 + 120);
            v82 = v80;
            v109 = &v80[2 * v81];
            if ( v80 == v109 )
            {
LABEL_155:
              v86 = 0;
              if ( (_DWORD)v81 )
              {
                do
                {
                  v88 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v79 + 112) + 16 * v86));
                  v110 = v88;
                  v89 = v88.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (v88.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == v73 )
                  {
                    ++v75;
                  }
                  else
                  {
                    v110.m128i_i64[0] = v79 | v88.m128i_i8[0] & 7;
                    nullsub_752(v74[11] + 824LL, v88.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL, v79 & 0xFFFFFFFFFFFFFFF8LL);
                    sub_1F01C30(v89);
                    sub_1F03360(v74[11] + 824LL, v73, v110.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
                    sub_1F01A00(v73, &v110, 1);
                    v87 = v74[11] + 824LL;
                    v110.m128i_i64[0] = v73 | v110.m128i_i8[0] & 7;
                    sub_1F03360(v87, v89, v110.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL);
                    sub_1F01A00(v89, &v110, 1);
                  }
                  v86 = v75;
                }
                while ( *(_DWORD *)(v79 + 120) != v75 );
              }
            }
            else
            {
              while ( 1 )
              {
                v83 = *v82 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v83 != v73 )
                {
                  if ( !*(_DWORD *)(v83 + 204) )
                    break;
                  if ( *(char *)(v73 + 228) < 0 && (*(_BYTE *)(v83 + 228) & 0x40) != 0 )
                  {
                    v105 = *v82 & 0xFFFFFFFFFFFFFFF8LL;
                    v84 = sub_1D02A20(*(_QWORD *)v83, *(_QWORD *)v73, v74[8], v74[9]);
                    v83 = v105;
                    if ( v84 )
                      break;
                  }
                  if ( (unsigned __int8)sub_1F03240(v74[11] + 824LL, v73, v83) )
                    break;
                }
                v82 += 2;
                if ( v109 == v82 )
                {
                  LODWORD(v81) = *(_DWORD *)(v79 + 120);
                  goto LABEL_155;
                }
              }
            }
          }
        }
      }
      v73 += 272;
    }
    while ( v106 != v73 );
    a2 = (__int64 *)v74[6];
    v2 = v74;
    v106 = a2[1];
  }
  v110.m128i_i32[0] = 0;
  sub_1D05C60((__int64)(v2 + 12), 0xF0F0F0F0F0F0F0F1LL * ((v106 - *a2) >> 4), v110.m128i_i32);
  v3 = (__int64 *)v2[6];
  v4 = *v3;
  v5 = v3[1];
  if ( *v3 != v5 )
  {
    do
    {
      while ( *(_DWORD *)(v2[12] + 4LL * *(unsigned int *)(v4 + 192)) )
      {
        v4 += 272;
        if ( v5 == v4 )
          goto LABEL_10;
      }
      v6 = v4;
      v4 += 272;
      sub_1D01FD0(v6, v2 + 12);
    }
    while ( v5 != v4 );
  }
LABEL_10:
  result = sub_1DD6970(*(_QWORD *)(v2[11] + 616LL), *(_QWORD *)(v2[11] + 616LL));
  v8 = result;
  if ( (_BYTE)result )
  {
    result = (__int64)v99;
    v9 = *v99;
    v10 = v99[1];
    if ( *v99 != v10 )
    {
      v11 = byte_4FC1100;
      do
      {
        while ( 1 )
        {
          if ( !v11 )
          {
            v12 = *(__int64 **)(v9 + 32);
            v13 = &v12[2 * *(unsigned int *)(v9 + 40)];
            if ( v12 != v13 )
            {
              v14 = *(__int64 **)(v9 + 32);
              v15 = 0;
              do
              {
                while ( 1 )
                {
                  result = *v14;
                  if ( (*v14 & 6) == 0 )
                    break;
                  v14 += 2;
                  if ( v13 == v14 )
                    goto LABEL_22;
                }
                result = *(_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL);
                if ( !result )
                  goto LABEL_16;
                if ( *(_WORD *)(result + 24) != 47 )
                  goto LABEL_16;
                result = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(result + 32) + 40LL) + 84LL);
                if ( (int)result >= 0 )
                  goto LABEL_16;
                v14 += 2;
                v15 = v8;
              }
              while ( v13 != v14 );
LABEL_22:
              if ( v15 )
              {
                result = sub_1D00EC0(v9);
                if ( (_BYTE)result )
                  break;
              }
            }
          }
LABEL_16:
          v9 += 272;
          if ( v10 == v9 )
            return result;
        }
        *(_BYTE *)(v9 + 228) |= 1u;
        do
        {
          result = *v12;
          if ( (*v12 & 6) == 0 )
          {
            result &= 0xFFFFFFFFFFFFFFF8LL;
            *(_BYTE *)(result + 228) |= 1u;
          }
          v12 += 2;
        }
        while ( v16 != v12 );
        v9 += 272;
      }
      while ( v10 != v9 );
    }
  }
  return result;
}
