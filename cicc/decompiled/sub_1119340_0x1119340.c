// Function: sub_1119340
// Address: 0x1119340
//
__int64 __fastcall sub_1119340(const __m128i *a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  int v4; // r13d
  unsigned __int8 *v5; // r15
  __int64 v7; // r8
  int v8; // edx
  __int64 v11; // rax
  _QWORD *v12; // r14
  bool v14; // al
  unsigned __int8 v15; // dl
  bool v16; // al
  unsigned __int8 v17; // dl
  unsigned __int64 v18; // r8
  int v19; // eax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r12
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  unsigned int v29; // r15d
  __int64 v30; // rax
  _BYTE *v31; // rsi
  _BYTE *v32; // rsi
  __int64 v33; // r9
  int v34; // r11d
  __int64 v35; // rax
  __m128i v36; // xmm1
  unsigned __int64 v37; // xmm2_8
  __m128i v38; // xmm3
  __int64 v39; // rax
  char v40; // al
  int v41; // r11d
  __int64 v42; // rax
  char v43; // al
  int v44; // r11d
  __int64 v45; // rsi
  unsigned int **v46; // rdi
  __int64 v47; // rax
  unsigned int **v48; // r15
  __int64 v49; // r14
  _BYTE *v50; // rax
  __int64 v51; // rax
  int v52; // r11d
  unsigned int **v53; // r14
  const char *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // r12
  _QWORD *v60; // rax
  _BYTE *v61; // rax
  unsigned int **v62; // rdi
  __int64 v63; // r12
  __int64 v64; // r15
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  char v67; // al
  char v68; // al
  unsigned int **v69; // r14
  _BYTE *v70; // rax
  __int64 v71; // rax
  int v72; // r11d
  _QWORD *v73; // rax
  char v74; // al
  __int64 v75; // rax
  unsigned __int8 v76; // cl
  __int64 *v77; // rdi
  char v78; // al
  unsigned int *v79; // r15
  __int64 v80; // r14
  __int64 v81; // rdx
  __int64 v82; // rdi
  __int64 v83; // rbx
  __int64 v84; // r14
  __int64 v85; // r12
  _QWORD *v86; // rax
  int v87; // eax
  unsigned int **v88; // rdi
  __int64 v89; // r14
  __int64 v90; // rbx
  __int64 v91; // r14
  __int64 v92; // rdx
  unsigned int v93; // esi
  int v94; // [rsp+10h] [rbp-100h]
  unsigned __int64 v95; // [rsp+18h] [rbp-F8h]
  _BYTE *v96; // [rsp+18h] [rbp-F8h]
  int v97; // [rsp+20h] [rbp-F0h]
  unsigned __int8 v98; // [rsp+20h] [rbp-F0h]
  __int64 v99; // [rsp+20h] [rbp-F0h]
  __int64 v100; // [rsp+20h] [rbp-F0h]
  __int64 v101; // [rsp+20h] [rbp-F0h]
  _BYTE *v102; // [rsp+28h] [rbp-E8h]
  int v103; // [rsp+28h] [rbp-E8h]
  int v104; // [rsp+28h] [rbp-E8h]
  int v105; // [rsp+28h] [rbp-E8h]
  int v106; // [rsp+28h] [rbp-E8h]
  char v107; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v108; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v109; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v110; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v111; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v112; // [rsp+58h] [rbp-B8h] BYREF
  _QWORD v113[4]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v114; // [rsp+80h] [rbp-90h]
  __m128i v115; // [rsp+90h] [rbp-80h] BYREF
  __m128i v116; // [rsp+A0h] [rbp-70h]
  __int64 *v117; // [rsp+B0h] [rbp-60h]
  __int64 v118; // [rsp+B8h] [rbp-58h]
  __m128i v119; // [rsp+C0h] [rbp-50h]
  __int64 v120; // [rsp+D0h] [rbp-40h]

  v4 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( (unsigned int)(v4 - 32) > 1 )
    return 0;
  v5 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  v7 = *(_QWORD *)(a2 - 32);
  v102 = (_BYTE *)*((_QWORD *)a3 - 8);
  v8 = *a3;
  switch ( v8 )
  {
    case '*':
      if ( *v5 <= 0x15u )
      {
        v25 = *((_QWORD *)a3 + 2);
        if ( v25 && !*(_QWORD *)(v25 + 8) )
        {
          v22 = sub_AD57F0(v7, v5, 0, 0);
          goto LABEL_22;
        }
        return 0;
      }
      if ( !sub_9867B0(a4) )
        return 0;
      v100 = sub_F0E5E0((__int64)a1, (__int64)v5);
      if ( v100 )
      {
        LOWORD(v117) = 257;
        v66 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v66;
        if ( v66 )
          sub_1113300((__int64)v66, v4, (__int64)v102, v100, (__int64)&v115);
        return (__int64)v12;
      }
      v101 = sub_F0E5E0((__int64)a1, (__int64)v102);
      if ( v101 )
      {
        LOWORD(v117) = 257;
        v73 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v73;
        if ( v73 )
          sub_1113300((__int64)v73, v4, v101, (__int64)v5, (__int64)&v115);
        return (__int64)v12;
      }
      v75 = *((_QWORD *)a3 + 2);
      if ( !v75 || *(_QWORD *)(v75 + 8) )
        return 0;
      v76 = *a3;
      if ( *a3 <= 0x36u && ((0x40540000000000uLL >> v76) & 1) != 0 )
      {
        v87 = v76 - 29;
        if ( v76 <= 0x1Cu )
          v87 = *((unsigned __int16 *)a3 + 1);
        if ( v87 == 13 && (a3[1] & 2) != 0 )
        {
          v88 = (unsigned int **)a1[2].m128i_i64[0];
          LOWORD(v117) = 257;
          v64 = sub_A82480(v88, v102, v5, (__int64)&v115);
          v63 = sub_AD6530(*((_QWORD *)a3 + 1), (__int64)v102);
          goto LABEL_62;
        }
      }
      v77 = (__int64 *)a1[2].m128i_i64[0];
      LOWORD(v117) = 257;
      v5 = sub_10A0530(v77, (__int64)v5, (__int64)&v115, 0);
      sub_BD6B90(v5, a3);
      LOWORD(v117) = 257;
      v12 = sub_BD2C40(72, unk_3F10FD0);
      if ( !v12 )
        return (__int64)v12;
      goto LABEL_75;
    case '0':
    case '1':
      v98 = v8;
      v95 = *(_QWORD *)(a2 - 32);
      v14 = sub_B44E60((__int64)a3);
      v15 = v98;
      if ( !v14 )
        goto LABEL_16;
      v16 = sub_9867B0(a4);
      v17 = v98;
      v18 = v95;
      if ( v16 )
      {
        v22 = sub_AD6530(*((_QWORD *)a3 + 1), a2);
        goto LABEL_22;
      }
      if ( *(_DWORD *)(a4 + 8) <= 0x40u )
      {
        if ( *(_QWORD *)a4 == 1 )
          goto LABEL_74;
      }
      else
      {
        v94 = *(_DWORD *)(a4 + 8);
        v19 = sub_C444A0(a4);
        v17 = v98;
        v18 = v95;
        if ( v19 == v94 - 1 )
          goto LABEL_74;
      }
      v20 = *((_QWORD *)a3 + 2);
      if ( !v20 || *(_QWORD *)(v20 + 8) )
        return 0;
      if ( (unsigned int)sub_1117350(a1, 17, v17 == 49, (__int64)v5, v18, (__int64)a3) == 3 )
      {
        v82 = *((_QWORD *)a3 + 1);
        v83 = a1[2].m128i_i64[0];
        v114 = 257;
        v84 = sub_AD8D80(v82, a4);
        v85 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64, _QWORD, _QWORD))(**(_QWORD **)(v83 + 80) + 32LL))(
                *(_QWORD *)(v83 + 80),
                17,
                v5,
                v84,
                0,
                0);
        if ( !v85 )
        {
          LOWORD(v117) = 257;
          v85 = sub_B504D0(17, (__int64)v5, v84, (__int64)&v115, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v83 + 88) + 16LL))(
            *(_QWORD *)(v83 + 88),
            v85,
            v113,
            *(_QWORD *)(v83 + 56),
            *(_QWORD *)(v83 + 64));
          v89 = 16LL * *(unsigned int *)(v83 + 8);
          v90 = *(_QWORD *)v83;
          v91 = v90 + v89;
          while ( v91 != v90 )
          {
            v92 = *(_QWORD *)(v90 + 8);
            v93 = *(_DWORD *)v90;
            v90 += 16;
            sub_B99FD0(v85, v93, v92);
          }
        }
        LOWORD(v117) = 257;
        v86 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v86;
        if ( v86 )
          sub_1113300((__int64)v86, v4, v85, (__int64)v102, (__int64)&v115);
        return (__int64)v12;
      }
      v15 = *a3;
LABEL_16:
      if ( v15 == 48 && sub_9867B0(a4) )
      {
        LOWORD(v117) = 257;
        v21 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v21;
        if ( v21 )
          sub_1113300((__int64)v21, 3 * (v4 == 33) + 34, (__int64)v5, (__int64)v102, (__int64)&v115);
        return (__int64)v12;
      }
      return 0;
    case '4':
      if ( *(_DWORD *)(a4 + 8) <= 0x40u )
      {
        if ( *(_QWORD *)a4 )
          return 0;
      }
      else
      {
        v97 = *(_DWORD *)(a4 + 8);
        if ( v97 != (unsigned int)sub_C444A0(a4) )
          return 0;
      }
      v11 = *((_QWORD *)a3 + 2);
      if ( !v11 )
        return 0;
      v12 = *(_QWORD **)(v11 + 8);
      if ( v12 )
        return 0;
      v115.m128i_i8[8] = 0;
      v115.m128i_i64[0] = (__int64)&v112;
      if ( (unsigned __int8)sub_991580((__int64)&v115, (__int64)v5) )
      {
        v99 = v112;
        if ( sub_AAD930(v112, 1) && sub_986BA0(v99) )
        {
          v53 = (unsigned int **)a1[2].m128i_i64[0];
          v54 = sub_BD5D20((__int64)a3);
          v55 = 22;
          v114 = 261;
          v113[1] = v56;
          v113[0] = v54;
          v57 = (*(__int64 (__fastcall **)(unsigned int *, __int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v53[10]
                                                                                               + 16LL))(
                  v53[10],
                  22,
                  v102,
                  v5);
          if ( !v57 )
          {
            LOWORD(v117) = 257;
            v57 = sub_B504D0(22, (__int64)v102, (__int64)v5, (__int64)&v115, 0, 0);
            v55 = v57;
            (*(void (__fastcall **)(unsigned int *, __int64, _QWORD *, unsigned int *, unsigned int *))(*(_QWORD *)v53[11] + 16LL))(
              v53[11],
              v57,
              v113,
              v53[7],
              v53[8]);
            v79 = *v53;
            v80 = (__int64)&(*v53)[4 * *((unsigned int *)v53 + 2)];
            while ( (unsigned int *)v80 != v79 )
            {
              v81 = *((_QWORD *)v79 + 1);
              v55 = *v79;
              v79 += 4;
              sub_B99FD0(v57, v55, v81);
            }
          }
          v58 = sub_AD6530(*((_QWORD *)a3 + 1), v55);
          LOWORD(v117) = 257;
          v59 = v58;
          v60 = sub_BD2C40(72, unk_3F10FD0);
          v12 = v60;
          if ( v60 )
            sub_1113300((__int64)v60, v4, v57, v59, (__int64)&v115);
        }
      }
      return (__int64)v12;
    case ':':
      v115.m128i_i64[0] = (__int64)&v107;
      v96 = (_BYTE *)v7;
      v115.m128i_i8[8] = 0;
      if ( (unsigned __int8)sub_991580((__int64)&v115, (__int64)v5) )
      {
        v28 = *((_QWORD *)a3 + 2);
        if ( v28 )
        {
          if ( !*(_QWORD *)(v28 + 8) && sub_AD7930(v96, (__int64)v5, v26, v27, (__int64)v96) )
          {
            v61 = (_BYTE *)sub_AD63D0((__int64)v5);
            v62 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v117) = 257;
            v63 = (__int64)v61;
            v64 = sub_A82350(v62, v102, v61, (__int64)&v115);
LABEL_62:
            LOWORD(v117) = 257;
            v65 = sub_BD2C40(72, unk_3F10FD0);
            v12 = v65;
            if ( v65 )
              sub_1113300((__int64)v65, v4, v64, v63, (__int64)&v115);
            return (__int64)v12;
          }
        }
      }
      v29 = *(_DWORD *)(a4 + 8);
      if ( v29 <= 0x40 )
      {
        if ( *(_QWORD *)a4 )
          return 0;
      }
      else if ( v29 != (unsigned int)sub_C444A0(a4) )
      {
        return 0;
      }
      v115.m128i_i64[0] = (__int64)&v112;
      v115.m128i_i64[1] = (__int64)&v108;
      v116.m128i_i64[0] = (__int64)&v109;
      v116.m128i_i64[1] = (__int64)&v110;
      v117 = &v111;
      v30 = *((_QWORD *)a3 + 2);
      if ( !v30 || *(_QWORD *)(v30 + 8) || *a3 != 58 )
        return 0;
      v31 = (_BYTE *)*((_QWORD *)a3 - 8);
      if ( v31
        && (v112 = *((_QWORD *)a3 - 8), *v31 == 86)
        && (unsigned __int8)sub_11110C0((_QWORD **)&v115.m128i_i64[1], (__int64)v31) )
      {
        v35 = *((_QWORD *)a3 - 4);
        if ( !v35 )
          return 0;
      }
      else
      {
        v32 = (_BYTE *)*((_QWORD *)a3 - 4);
        if ( !v32 )
          return 0;
        *(_QWORD *)v115.m128i_i64[0] = v32;
        if ( *v32 != 86 )
          return 0;
        if ( !(unsigned __int8)sub_11110C0((_QWORD **)&v115.m128i_i64[1], (__int64)v32) )
          return 0;
        v35 = *((_QWORD *)a3 - 8);
        if ( !v35 )
          return 0;
      }
      *v117 = v35;
      if ( *(_QWORD *)(v33 + 8) != *(_QWORD *)(v108 + 8) )
        return 0;
      v36 = _mm_loadu_si128(a1 + 7);
      v103 = v34;
      v37 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v38 = _mm_loadu_si128(a1 + 9);
      v39 = a1[10].m128i_i64[0];
      v115 = _mm_loadu_si128(a1 + 6);
      v117 = (__int64 *)v37;
      v120 = v39;
      v118 = v33;
      v116 = v36;
      v119 = v38;
      if ( v4 == 32 )
      {
        v67 = sub_1112510(v109);
        v41 = v103;
        if ( !v67 )
          goto LABEL_44;
        v68 = sub_9B6260(v110, &v115, 0);
        v41 = v103;
        if ( !v68 )
          goto LABEL_44;
LABEL_80:
        v69 = (unsigned int **)a1[2].m128i_i64[0];
        v106 = v41;
        v114 = 257;
        v70 = (_BYTE *)sub_AD6530(*(_QWORD *)(v111 + 8), 257);
        v71 = sub_92B530(v69, v4, v111, v70, (__int64)v113);
        v72 = v106;
        if ( v4 == 32 )
          v72 = 28;
        v114 = 257;
        return sub_B504D0(v72, v71, v108, (__int64)v113, 0, 0);
      }
      v40 = sub_1112510(v110);
      v41 = v103;
      if ( v40 )
      {
        v74 = sub_9B6260(v109, &v115, 0);
        v41 = v103;
        if ( v74 )
          goto LABEL_80;
      }
LABEL_44:
      v42 = *(_QWORD *)(v112 + 16);
      if ( !v42 || *(_QWORD *)(v42 + 8) )
        return 0;
      v104 = v41;
      if ( v4 == 32 )
      {
        if ( !(unsigned __int8)sub_1112510(v110) )
          return 0;
        v78 = sub_9B6260(v109, &v115, 0);
        v44 = v104;
        if ( !v78 )
          return 0;
      }
      else
      {
        if ( !(unsigned __int8)sub_1112510(v109) )
          return 0;
        v43 = sub_9B6260(v110, &v115, 0);
        v44 = v104;
        if ( !v43 )
          return 0;
      }
      v45 = v108;
      v46 = (unsigned int **)a1[2].m128i_i64[0];
      v105 = v44;
      v114 = 257;
      v47 = sub_A82B60(v46, v108, (__int64)v113);
      v48 = (unsigned int **)a1[2].m128i_i64[0];
      v114 = 257;
      v49 = v47;
      v50 = (_BYTE *)sub_AD6530(*(_QWORD *)(v111 + 8), v45);
      v51 = sub_92B530(v48, v4, v111, v50, (__int64)v113);
      v52 = v105;
      v114 = 257;
      if ( v4 == 32 )
        v52 = 28;
      return sub_B504D0(v52, v51, v49, (__int64)v113, 0, 0);
    case ';':
      if ( *v5 > 0x15u )
      {
        if ( !sub_9867B0(a4) )
          return 0;
LABEL_74:
        LOWORD(v117) = 257;
        v12 = sub_BD2C40(72, unk_3F10FD0);
        if ( v12 )
LABEL_75:
          sub_1113300((__int64)v12, v4, (__int64)v102, (__int64)v5, (__int64)&v115);
      }
      else
      {
        v22 = sub_AD5820(v7, v5);
LABEL_22:
        v23 = v22;
        LOWORD(v117) = 257;
        v24 = sub_BD2C40(72, unk_3F10FD0);
        v12 = v24;
        if ( v24 )
          sub_1113300((__int64)v24, v4, (__int64)v102, v23, (__int64)&v115);
      }
      return (__int64)v12;
    default:
      return 0;
  }
}
