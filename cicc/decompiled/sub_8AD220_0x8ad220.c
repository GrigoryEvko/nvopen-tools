// Function: sub_8AD220
// Address: 0x8ad220
//
__int64 __fastcall sub_8AD220(__int64 a1, int *a2)
{
  __int64 *v2; // r15
  __int64 v3; // r12
  __int64 v4; // r14
  bool v5; // bl
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 result; // rax
  int v9; // r8d
  int v10; // r8d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v16; // r10
  __int64 v17; // rcx
  __int64 v18; // r10
  __int64 v19; // r9
  __m128i *v20; // rax
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rsi
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  const __m128i *v34; // r14
  __int64 v35; // rdi
  unsigned __int16 v36; // dx
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm0
  __m128i v40; // xmm1
  __m128i v41; // xmm2
  __m128i v42; // xmm3
  __m128i v43; // xmm4
  __m128i v44; // xmm5
  __m128i v45; // xmm6
  __m128i v46; // xmm7
  __m128i v47; // xmm0
  __m128i v48; // xmm1
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  __m128i v51; // xmm6
  __m128i v52; // xmm7
  __m128i v53; // xmm0
  __m128i v54; // xmm1
  __m128i v55; // xmm2
  __m128i v56; // xmm3
  __m128i v57; // xmm4
  __m128i v58; // xmm5
  __m128i v59; // xmm6
  __m128i v60; // xmm7
  __int64 v61; // rax
  size_t v62; // rax
  size_t v63; // rax
  unsigned int v64; // [rsp+8h] [rbp-418h]
  __int64 v65; // [rsp+10h] [rbp-410h]
  __int64 v66; // [rsp+10h] [rbp-410h]
  int v67; // [rsp+1Ch] [rbp-404h]
  unsigned int v68; // [rsp+20h] [rbp-400h]
  __int64 v69; // [rsp+20h] [rbp-400h]
  __int64 v70; // [rsp+20h] [rbp-400h]
  __int64 v71; // [rsp+28h] [rbp-3F8h]
  __int64 v72; // [rsp+28h] [rbp-3F8h]
  __int64 v73; // [rsp+28h] [rbp-3F8h]
  unsigned int v74; // [rsp+28h] [rbp-3F8h]
  _BOOL4 v75; // [rsp+30h] [rbp-3F0h]
  unsigned int v76; // [rsp+34h] [rbp-3ECh]
  _QWORD *v77; // [rsp+38h] [rbp-3E8h]
  int v78; // [rsp+40h] [rbp-3E0h]
  __int16 v79; // [rsp+46h] [rbp-3DAh]
  int v80; // [rsp+48h] [rbp-3D8h]
  unsigned __int16 v81; // [rsp+4Ch] [rbp-3D4h]
  unsigned __int16 v82; // [rsp+4Eh] [rbp-3D2h]
  char *v83; // [rsp+50h] [rbp-3D0h]
  int v84; // [rsp+58h] [rbp-3C8h]
  __int64 v85; // [rsp+58h] [rbp-3C8h]
  __int64 v86; // [rsp+60h] [rbp-3C0h]
  __int64 v87; // [rsp+60h] [rbp-3C0h]
  char *s; // [rsp+68h] [rbp-3B8h]
  char *sa; // [rsp+68h] [rbp-3B8h]
  __m128i v90; // [rsp+70h] [rbp-3B0h] BYREF
  __m128i v91; // [rsp+80h] [rbp-3A0h] BYREF
  __m128i v92; // [rsp+90h] [rbp-390h] BYREF
  __m128i v93; // [rsp+A0h] [rbp-380h] BYREF
  __m128i v94; // [rsp+B0h] [rbp-370h] BYREF
  __m128i v95; // [rsp+C0h] [rbp-360h] BYREF
  __m128i v96; // [rsp+D0h] [rbp-350h] BYREF
  __m128i v97; // [rsp+E0h] [rbp-340h] BYREF
  __m128i v98; // [rsp+F0h] [rbp-330h] BYREF
  __m128i v99; // [rsp+100h] [rbp-320h] BYREF
  __m128i v100; // [rsp+110h] [rbp-310h] BYREF
  __m128i v101; // [rsp+120h] [rbp-300h] BYREF
  __m128i v102; // [rsp+130h] [rbp-2F0h] BYREF
  __m128i v103; // [rsp+140h] [rbp-2E0h] BYREF
  __m128i v104; // [rsp+150h] [rbp-2D0h] BYREF
  __m128i v105; // [rsp+160h] [rbp-2C0h] BYREF
  __m128i v106; // [rsp+170h] [rbp-2B0h] BYREF
  __m128i v107; // [rsp+180h] [rbp-2A0h] BYREF
  __m128i v108; // [rsp+190h] [rbp-290h] BYREF
  __m128i v109; // [rsp+1A0h] [rbp-280h] BYREF
  __m128i v110; // [rsp+1B0h] [rbp-270h] BYREF
  __m128i v111; // [rsp+1C0h] [rbp-260h] BYREF
  __m128i v112; // [rsp+1D0h] [rbp-250h] BYREF
  __m128i v113; // [rsp+1E0h] [rbp-240h] BYREF
  __m128i v114; // [rsp+1F0h] [rbp-230h] BYREF
  __m128i v115; // [rsp+200h] [rbp-220h] BYREF
  _QWORD v116[66]; // [rsp+210h] [rbp-210h] BYREF

  v2 = (__int64 *)a1;
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    do
      v2 = (__int64 *)v2[20];
    while ( *((_BYTE *)v2 + 140) == 12 );
  }
  v3 = *v2;
  v4 = *(_QWORD *)(*v2 + 96);
  v5 = (*((_BYTE *)v2 + 89) & 4) != 0;
  *(_QWORD *)(v4 + 120) = sub_893360();
  v6 = sub_878920(v3);
  v7 = v6;
  if ( v6 )
  {
    switch ( *(_BYTE *)(v6 + 80) )
    {
      case 4:
      case 5:
        s = *(char **)(*(_QWORD *)(v6 + 96) + 80LL);
        break;
      case 6:
        s = *(char **)(*(_QWORD *)(v6 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        s = *(char **)(*(_QWORD *)(v6 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        s = *(char **)(v6 + 88);
        break;
      default:
        BUG();
    }
    if ( *((_QWORD *)s + 13) )
    {
      if ( (unsigned int)sub_825090() )
      {
        result = sub_8250A0();
        if ( (*((_BYTE *)v2 + 141) & 0x20) == 0 )
          return result;
      }
      else if ( (unsigned int)sub_8250B0() )
      {
        result = sub_8250C0();
        if ( (*((_BYTE *)v2 + 141) & 0x20) == 0 )
          return result;
      }
    }
    result = *((_BYTE *)v2 + 177) & 0x60;
    if ( (_BYTE)result == 32 )
      return result;
    result = *(unsigned __int8 *)(v4 + 180);
    if ( (result & 8) != 0 || (*((_BYTE *)v2 + 178) & 1) != 0 )
      return result;
    v77 = qword_4D03CB8;
    v79 = dword_4F07508[1];
    qword_4D03CB8 = 0;
    v76 = dword_4F063F8;
    v81 = word_4F063FC[0];
    v78 = dword_4F07508[0];
    v80 = dword_4F061D8;
    v82 = word_4F061DC[0];
    *(_BYTE *)(v4 + 180) = result | 8;
    if ( (unsigned __int16)(word_4F06418[0] - 2) <= 6u )
    {
      v90 = _mm_loadu_si128(xmmword_4F06300);
      v91 = _mm_loadu_si128(&xmmword_4F06300[1]);
      v92 = _mm_loadu_si128(&xmmword_4F06300[2]);
      v93 = _mm_loadu_si128(&xmmword_4F06300[3]);
      v94 = _mm_loadu_si128(&xmmword_4F06300[4]);
      v95 = _mm_loadu_si128(&xmmword_4F06300[5]);
      v96 = _mm_loadu_si128(&xmmword_4F06300[6]);
      v97 = _mm_loadu_si128(&xmmword_4F06300[7]);
      v98 = _mm_loadu_si128(xmmword_4F06380);
      v99 = _mm_loadu_si128(&xmmword_4F06380[1]);
      v100 = _mm_loadu_si128((const __m128i *)&unk_4F063A0);
      v101 = _mm_loadu_si128((const __m128i *)word_4F063B0);
      v102 = _mm_loadu_si128(&xmmword_4F063C0);
      if ( word_4F06418[0] == 8 )
      {
        v103 = _mm_loadu_si128(xmmword_4F06220);
        v104 = _mm_loadu_si128(&xmmword_4F06220[1]);
        v105 = _mm_loadu_si128(&xmmword_4F06220[2]);
        v106 = _mm_loadu_si128(&xmmword_4F06220[3]);
        v107 = _mm_loadu_si128(&xmmword_4F06220[4]);
        v108 = _mm_loadu_si128(&xmmword_4F06220[5]);
        v109 = _mm_loadu_si128(&xmmword_4F06220[6]);
        v110 = _mm_loadu_si128(&xmmword_4F06220[7]);
        v111 = _mm_loadu_si128(xmmword_4F062A0);
        v112 = _mm_loadu_si128(&xmmword_4F062A0[1]);
        v113 = _mm_loadu_si128((const __m128i *)&unk_4F062C0);
        v114 = _mm_loadu_si128((const __m128i *)&unk_4F062D0);
        v115 = _mm_loadu_si128(xmmword_4F062E0);
      }
    }
    v75 = (*((_BYTE *)v2 + 177) & 0x40) != 0;
    v9 = (*((_BYTE *)v2 + 177) & 0x40) != 0 ? 4 : 0;
    if ( (s[160] & 1) != 0 )
      v9 |= 0x400000u;
    v84 = v9;
    LODWORD(v11) = sub_8D0B70(v7);
    v10 = v84;
    v67 = v11;
    LOBYTE(v11) = *(_BYTE *)(v7 + 80);
    v83 = s;
    if ( (_BYTE)v11 == 19 )
    {
      if ( !*((_QWORD *)s + 18) )
        goto LABEL_19;
      v11 = sub_8A9D50(v3, *(_QWORD *)(v7 + 88), a2);
      v10 = v84;
      v13 = v11;
      if ( v11 )
      {
        LOBYTE(v11) = *(_BYTE *)(v11 + 80);
        switch ( (char)v11 )
        {
          case 4:
          case 5:
            v7 = v13;
            v83 = *(char **)(*(_QWORD *)(v13 + 96) + 80LL);
            goto LABEL_18;
          case 6:
            v7 = v13;
            v83 = *(char **)(*(_QWORD *)(v13 + 96) + 32LL);
LABEL_87:
            v85 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 32LL);
            goto LABEL_24;
          case 9:
          case 10:
            v7 = v13;
            v83 = *(char **)(*(_QWORD *)(v13 + 96) + 56LL);
LABEL_85:
            v85 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 56LL);
LABEL_24:
            v68 = v10;
            v71 = *(_QWORD *)(v85 + 176);
            v14 = sub_892400(v85);
            v15 = v68;
            v16 = v14;
            if ( !a2 || !*a2 )
              s[266] |= 8u;
            *(_QWORD *)(v4 + 104) = v71;
            if ( *(_QWORD *)(v14 + 8) && (*(_BYTE *)(v85 + 265) & 2) != 0 && (*(_BYTE *)(v4 + 182) & 1) == 0 )
            {
              if ( (unsigned __int64)*((unsigned int *)v83 + 10) >= unk_4D042F0 )
              {
                sub_6854E0(0x1C8u, v3);
                *((_BYTE *)v2 + 178) |= 1u;
              }
              else
              {
                v17 = *(_QWORD *)(v71 + 88);
                v86 = v17;
                if ( *(_QWORD *)(*(_QWORD *)(v71 + 96) + 56LL) )
                {
                  v72 = v14;
                  sub_5EAAC0(v17, 1u, 0);
                  sub_5EAAC0(v86, 1u, 1u);
                  v16 = v72;
                  v15 = v68;
                }
                *(_QWORD *)(v4 + 88) = *(_QWORD *)&dword_4F063F8;
                ++*((_DWORD *)v83 + 10);
                if ( *(_BYTE *)(v7 + 80) == 19 )
                  *(_QWORD *)(v4 + 72) = v7;
                sa = 0;
                v69 = v2[21];
                *((_BYTE *)v2 + 140) = v83[264];
                *(_QWORD *)(v69 + 160) = *(_QWORD *)(v85 + 104);
                if ( dword_4D0460C )
                {
                  v66 = v16;
                  v74 = v15;
                  sa = (char *)_libc_calloc(300000, 1);
                  sub_87D390((__int64)sa, v3, 299997, 1);
                  v62 = strlen(sa);
                  sa[v62] = 32;
                  sa[v62 + 1] = 91;
                  sub_87D390((__int64)&sa[v62 + 2], v3, 299997 - v62, 0);
                  v63 = strlen(sa);
                  v16 = v66;
                  v15 = v74;
                  sa[v63] = 93;
                  sa[v63 + 1] = 0;
                }
                v65 = v16;
                v64 = v15;
                unk_4D045D8("Instantiating Template Class", sa);
                v73 = sub_892330((__int64)v2);
                sub_864700(*(_QWORD *)(v65 + 32), (__int64)v2, 0, v3, v7, v73, 1, v64);
                v18 = v65;
                if ( *(_QWORD *)(v85 + 128) )
                {
                  memset(v116, 0, 0x1D8u);
                  v116[19] = v116;
                  v116[3] = *(_QWORD *)&dword_4F063F8;
                  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
                    BYTE2(v116[22]) |= 1u;
                  v19 = 0;
                  if ( (*((_BYTE *)v2 + 89) & 4) != 0 )
                    v19 = *(_QWORD *)(v2[5] + 32);
                  v20 = (__m128i *)sub_5CF220(
                                     *(const __m128i **)(v85 + 128),
                                     1,
                                     v7,
                                     **(_QWORD **)(v85 + 32),
                                     v73,
                                     v19,
                                     0,
                                     0);
                  sub_66A990(v20, (__int64)v2, (__int64)v116, 1, 0, 0);
                  v18 = v65;
                  if ( v2[13] )
                  {
                    sub_656C00((__int64)v116, 6, (__int64)v2, 0, 1);
                    v18 = v65;
                  }
                }
                v21 = *(_BYTE *)(v69 + 109);
                if ( (v21 & 7) == 0 )
                  *(_BYTE *)(v69 + 109) = *(_BYTE *)(*(_QWORD *)(v86 + 168) + 109LL) & 7 | v21 & 0xF8;
                v70 = v18;
                sub_854C10(*((const __m128i **)v83 + 7));
                sub_7BC160(v70);
                if ( (*(_BYTE *)(v86 + 176) & 1) != 0 )
                  *((_BYTE *)v2 + 176) |= 1u;
                *((_BYTE *)v2 + 141) = *(_BYTE *)(v86 + 141) & 0x10 | *((_BYTE *)v2 + 141) & 0xEF;
                sub_8756F0(32770, v3, (_QWORD *)(v3 + 48), 0);
                v22 = qword_4F04C68[0];
                ++*(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720);
                if ( dword_4F077C4 == 2 )
                  *(_BYTE *)(v22 + 776LL * (int)dword_4F04C40 + 7) |= 8u;
                v23 = 0;
                v24 = (unsigned __int64)v2;
                sub_607B60(v2, 0, dword_4F04C34, 0, 0, v5 & 0x3F, 1u, 0, 0, 0);
                v25 = qword_4F04C68[0];
                --*(_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C24 + 720);
                if ( dword_4F077C4 == 2 )
                {
                  v24 = (int)dword_4F04C40;
                  v61 = 776LL * (int)dword_4F04C40;
                  *(_BYTE *)(v25 + v61 + 7) &= ~8u;
                  if ( *(_QWORD *)(qword_4F04C68[0] + v61 + 456) )
                    sub_8845B0(v24);
                }
                if ( v75 )
                {
                  sub_854B40();
                }
                else
                {
                  v23 = 0;
                  v24 = v3;
                  sub_854980(v3, 0);
                }
                sub_863FE0(v24, 0, v26, v27, v28, v29);
                ++dword_4D03B6C;
                while ( word_4F06418[0] != 9 )
                  sub_7B8B50(v24, 0, v30, v31, v32, v33);
                sub_7B8B50(v24, 0, v30, v31, v32, v33);
                --*((_DWORD *)v83 + 10);
                if ( *(_QWORD *)(v85 + 184) )
                {
                  v87 = v4;
                  v34 = *(const __m128i **)(v85 + 184);
                  do
                  {
                    v23 = v3;
                    sub_8A9820(v34, v3, (__int64)v2, v7, v73);
                    v34 = (const __m128i *)v34->m128i_i64[0];
                  }
                  while ( v34 );
                  v4 = v87;
                }
                sub_8CF3D0(v2);
                if ( v83[160] >= 0 && !v75 )
                  sub_8AD0F0((__int64)v2);
                v35 = 1;
                sub_5F94C0(1);
                if ( v67 )
                  sub_8D0B10();
                if ( dword_4D0460C )
                {
                  v35 = (__int64)sa;
                  _libc_free(sa, v23);
                }
                unk_4D045D0(v35, v23);
              }
            }
            *(_BYTE *)(v4 + 180) &= ~8u;
            v36 = word_4F06418[0];
            if ( (unsigned __int16)(word_4F06418[0] - 2) <= 6u )
            {
              v37 = _mm_loadu_si128(&v91);
              v38 = _mm_loadu_si128(&v92);
              v39 = _mm_loadu_si128(&v93);
              v40 = _mm_loadu_si128(&v94);
              xmmword_4F06300[0] = _mm_loadu_si128(&v90);
              v41 = _mm_loadu_si128(&v95);
              v42 = _mm_loadu_si128(&v96);
              xmmword_4F06300[1] = v37;
              v43 = _mm_loadu_si128(&v97);
              v44 = _mm_loadu_si128(&v98);
              xmmword_4F06300[2] = v38;
              v45 = _mm_loadu_si128(&v99);
              v46 = _mm_loadu_si128(&v100);
              xmmword_4F06300[3] = v39;
              xmmword_4F06300[4] = v40;
              v47 = _mm_loadu_si128(&v101);
              v48 = _mm_loadu_si128(&v102);
              xmmword_4F06300[5] = v41;
              xmmword_4F06300[6] = v42;
              xmmword_4F06300[7] = v43;
              xmmword_4F06380[0] = v44;
              xmmword_4F06380[1] = v45;
              unk_4F063A0 = v46;
              *(__m128i *)word_4F063B0 = v47;
              xmmword_4F063C0 = v48;
              if ( v36 == 8 )
              {
                v49 = _mm_loadu_si128(&v104);
                v50 = _mm_loadu_si128(&v105);
                v51 = _mm_loadu_si128(&v106);
                v52 = _mm_loadu_si128(&v107);
                xmmword_4F06220[0] = _mm_loadu_si128(&v103);
                v53 = _mm_loadu_si128(&v108);
                v54 = _mm_loadu_si128(&v109);
                xmmword_4F06220[1] = v49;
                v55 = _mm_loadu_si128(&v110);
                v56 = _mm_loadu_si128(&v111);
                xmmword_4F06220[2] = v50;
                v57 = _mm_loadu_si128(&v112);
                v58 = _mm_loadu_si128(&v113);
                xmmword_4F06220[3] = v51;
                xmmword_4F06220[4] = v52;
                v59 = _mm_loadu_si128(&v114);
                v60 = _mm_loadu_si128(&v115);
                xmmword_4F06220[5] = v53;
                xmmword_4F06220[6] = v54;
                xmmword_4F06220[7] = v55;
                xmmword_4F062A0[0] = v56;
                xmmword_4F062A0[1] = v57;
                unk_4F062C0 = v58;
                unk_4F062D0 = v59;
                xmmword_4F062E0[0] = v60;
              }
            }
            dword_4F07508[0] = v78;
            LOWORD(dword_4F07508[1]) = v79;
            dword_4F063F8 = v76;
            word_4F063FC[0] = v81;
            dword_4F061D8 = v80;
            word_4F061DC[0] = v82;
            result = (__int64)&qword_4D03CB8;
            qword_4D03CB8 = v77;
            return result;
          case 19:
          case 20:
          case 21:
          case 22:
            v7 = v13;
            v83 = *(char **)(v13 + 88);
            goto LABEL_18;
          default:
            v83 = 0;
            v7 = v13;
            goto LABEL_18;
        }
      }
      v11 = *(unsigned __int8 *)(v7 + 80);
    }
LABEL_18:
    if ( (unsigned __int8)(v11 - 19) > 3u )
    {
      v13 = v7;
LABEL_22:
      switch ( (char)v11 )
      {
        case 4:
        case 5:
          v85 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 80LL);
          goto LABEL_24;
        case 6:
          goto LABEL_87;
        case 9:
        case 10:
          goto LABEL_85;
        case 19:
        case 20:
        case 21:
        case 22:
          v85 = *(_QWORD *)(v13 + 88);
          goto LABEL_24;
        default:
          BUG();
      }
    }
LABEL_19:
    v12 = *(_QWORD *)(v7 + 88);
    v13 = *(_QWORD *)(v12 + 88);
    if ( v13 && (*(_BYTE *)(v12 + 160) & 1) == 0 )
    {
      LOBYTE(v11) = *(_BYTE *)(v13 + 80);
    }
    else
    {
      LOBYTE(v11) = *(_BYTE *)(v7 + 80);
      v13 = v7;
    }
    goto LABEL_22;
  }
  result = sub_8250D0(v2);
  if ( (_DWORD)result )
    return sub_8250E0(v2);
  return result;
}
