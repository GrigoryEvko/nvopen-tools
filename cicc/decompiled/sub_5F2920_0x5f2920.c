// Function: sub_5F2920
// Address: 0x5f2920
//
__int64 __fastcall sub_5F2920(__int64 a1, __int64 *a2, __int64 a3, _BYTE *a4)
{
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdi
  unsigned int *v8; // rcx
  unsigned int v9; // ebx
  __int64 v10; // rbx
  __m128i *v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rsi
  char v14; // al
  __int8 v15; // al
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int16 v20; // ax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int8 v23; // si
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdi
  unsigned __int16 v28; // ax
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rcx
  int v34; // r10d
  unsigned int v35; // eax
  __int64 v36; // rax
  __m128i *v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rsi
  char v41; // al
  __int64 v42; // rcx
  __int64 v43; // r8
  char v44; // dl
  __int8 v45; // al
  __int64 v46; // rax
  _BOOL8 v47; // rcx
  int v48; // edx
  __int64 result; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // rax
  unsigned __int8 v56; // al
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // rax
  _DWORD *v61; // rax
  __int64 v62; // rax
  unsigned int v63; // eax
  __int64 v64; // rax
  __int64 v65; // rax
  const __m128i *v66; // rax
  int v67; // eax
  int v68; // eax
  __int64 v69; // [rsp+8h] [rbp-A8h]
  int v70; // [rsp+10h] [rbp-A0h]
  int v71; // [rsp+10h] [rbp-A0h]
  unsigned int v72; // [rsp+14h] [rbp-9Ch]
  _BYTE *v73; // [rsp+18h] [rbp-98h]
  __int64 v74; // [rsp+18h] [rbp-98h]
  __int64 v75; // [rsp+20h] [rbp-90h]
  __int64 v76; // [rsp+20h] [rbp-90h]
  int v77; // [rsp+28h] [rbp-88h]
  __int64 v78; // [rsp+28h] [rbp-88h]
  __int64 v79; // [rsp+30h] [rbp-80h]
  unsigned int v80; // [rsp+30h] [rbp-80h]
  __m128i *v81; // [rsp+38h] [rbp-78h]
  __int64 v82; // [rsp+38h] [rbp-78h]
  __int64 v83; // [rsp+38h] [rbp-78h]
  __m128i *v84; // [rsp+40h] [rbp-70h]
  __int64 v85; // [rsp+48h] [rbp-68h]
  __int64 v86; // [rsp+48h] [rbp-68h]
  unsigned __int8 v87; // [rsp+48h] [rbp-68h]
  unsigned int v88; // [rsp+48h] [rbp-68h]
  unsigned __int8 v89; // [rsp+48h] [rbp-68h]
  __int64 v90; // [rsp+48h] [rbp-68h]
  int v91; // [rsp+48h] [rbp-68h]
  int v92; // [rsp+48h] [rbp-68h]
  __int64 v94; // [rsp+58h] [rbp-58h]
  int v97; // [rsp+74h] [rbp-3Ch] BYREF
  _QWORD v98[7]; // [rsp+78h] [rbp-38h] BYREF

  v5 = *a2;
  v73 = a4 + 32;
  v6 = *((_QWORD *)a4 + 36);
  v75 = *((_QWORD *)a4 + 35);
  v72 = dword_4F06650[0];
  if ( (unsigned int)sub_8D2600(v6) )
  {
    v7 = (unsigned int)sub_67F240(v6);
    sub_685A50(v7, dword_4F07508, v6, 8);
    v6 = sub_72C930();
    goto LABEL_3;
  }
  if ( !(unsigned int)sub_8D5830(v6) )
  {
LABEL_3:
    if ( (*((_BYTE *)a2 + 9) & 4) == 0 )
      goto LABEL_4;
LABEL_50:
    sub_6851C0(817, v73);
    v6 = sub_72C930();
    goto LABEL_7;
  }
  sub_5EB950(8u, 322, v6, a1 + 8);
  if ( (*((_BYTE *)a2 + 9) & 4) != 0 )
    goto LABEL_50;
LABEL_4:
  if ( dword_4D04434 )
  {
    v8 = (unsigned int *)(HIDWORD(qword_4D0495C) | (unsigned int)qword_4D0495C);
    if ( (qword_4D0495C || dword_4F077BC) && *(_BYTE *)(v5 + 140) != 11 )
      goto LABEL_7;
    if ( (*(_BYTE *)(v5 + 177) & 4) != 0 )
    {
LABEL_227:
      v56 = unk_4F07470;
      if ( *(_BYTE *)(v5 + 140) == 11 )
      {
        v57 = 8;
        if ( dword_4D04434 )
          goto LABEL_230;
      }
      goto LABEL_229;
    }
    goto LABEL_54;
  }
  if ( *(_BYTE *)(v5 + 140) == 11 )
  {
    sub_6851C0(817, v73);
    goto LABEL_7;
  }
  v8 = (unsigned int *)(HIDWORD(qword_4D0495C) | (unsigned int)qword_4D0495C);
  if ( !qword_4D0495C && !dword_4F077BC )
  {
    if ( (*(_BYTE *)(v5 + 177) & 4) != 0 )
    {
      v56 = unk_4F07470;
LABEL_229:
      v57 = v56;
LABEL_230:
      sub_684AA0(v57, 817, v73);
      goto LABEL_7;
    }
LABEL_54:
    v24 = v5;
    while ( (*(_BYTE *)(v24 + 89) & 4) != 0 )
    {
      v24 = *(_QWORD *)(*(_QWORD *)(v24 + 40) + 32LL);
      if ( (*(_BYTE *)(v24 + 177) & 4) != 0 )
        goto LABEL_227;
    }
  }
LABEL_7:
  if ( (a4[10] & 8) != 0 && ((*(_BYTE *)(v6 + 140) & 0xFB) != 8 || (sub_8D4C10(v6, dword_4F077C4 != 2) & 1) == 0) )
    v6 = sub_73C570(v6, 1, -1);
  *((_QWORD *)a4 + 36) = v6;
  v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 168) + 152LL) + 240LL);
  if ( !unk_4D043C8 )
  {
    if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
      goto LABEL_12;
    v8 = &dword_4F077BC;
    if ( !dword_4F077BC )
    {
      if ( !dword_4F077B4 )
        goto LABEL_12;
      goto LABEL_69;
    }
    if ( dword_4F077B4 )
    {
LABEL_69:
      if ( unk_4F077A0 > 0x76BFu )
        goto LABEL_70;
LABEL_12:
      if ( (a4[561] & 2) != 0 )
      {
        *(_BYTE *)(a1 + 17) |= 0x20u;
        *(_QWORD *)(a1 + 24) = 0;
        if ( (a4[561] & 2) != 0 )
          goto LABEL_14;
      }
      goto LABEL_71;
    }
    if ( qword_4F077A8 <= 0xC34Fu )
      goto LABEL_12;
  }
LABEL_70:
  if ( (a4[561] & 2) != 0 )
    goto LABEL_76;
LABEL_71:
  v26 = sub_735FB0(v6, 2, 0xFFFFFFFFLL, v8);
  v11 = (__m128i *)v26;
  if ( (a4[8] & 2) != 0 )
    sub_658080(v26, 1);
  sub_735E40(v11, v9);
  if ( (a4[561] & 2) != 0 )
  {
LABEL_76:
    v27 = *(_QWORD *)(a1 + 24);
    if ( v27 && *(_BYTE *)(v27 + 80) == 7 && (*(_BYTE *)(a1 + 18) & 1) != 0 )
    {
      v10 = sub_89B100(v27, a3, a1, v8);
LABEL_15:
      v11 = *(__m128i **)(*(_QWORD *)(v10 + 88) + 192LL);
      v12 = dword_4F06650[0];
      v84 = *(__m128i **)(v10 + 88);
      v11[7].m128i_i64[1] = v6;
      sub_8975E0(a3, v12, 0);
      v13 = a3 + 288;
      sub_879080(&v84[12].m128i_u64[1], a3 + 288, *(_QWORD *)(a3 + 192));
      *(_DWORD *)(a3 + 320) = 1;
      v84[15].m128i_i32[0] = *((_DWORD *)a4 + 17);
      if ( (a4[8] & 2) != 0 )
      {
        v13 = 1;
        sub_658080(v11, 1);
      }
      if ( *(_BYTE *)(v10 + 80) != 21 )
        goto LABEL_75;
      goto LABEL_18;
    }
LABEL_14:
    v10 = sub_898DA0(a3, a1, 0, v8);
    goto LABEL_15;
  }
  v13 = a1;
  v84 = 0;
  v10 = sub_885AD0(9, a1, v9, 0);
  if ( *(_BYTE *)(v10 + 80) != 21 )
  {
LABEL_75:
    sub_877D80(v11, v10);
    *(_QWORD *)(v10 + 88) = v11;
    v13 = (__int64)v11;
    sub_877E20(v10, v11, v5);
    v85 = v10;
    goto LABEL_19;
  }
LABEL_18:
  v85 = v11->m128i_i64[0];
LABEL_19:
  *(_QWORD *)a4 = v10;
  if ( unk_4D043C8 )
    goto LABEL_23;
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774 )
    goto LABEL_21;
  if ( dword_4F077BC )
  {
    if ( !dword_4F077B4 )
    {
      if ( qword_4F077A8 > 0xC34Fu )
        goto LABEL_203;
      goto LABEL_21;
    }
  }
  else if ( !dword_4F077B4 )
  {
    goto LABEL_21;
  }
  if ( unk_4F077A0 <= 0x76BFu )
  {
LABEL_21:
    if ( (a4[561] & 2) != 0 )
    {
      v13 = a1 + 8;
      sub_6854C0(786, a1 + 8, v10);
      a4[561] |= 0x20u;
    }
LABEL_23:
    if ( (a4[10] & 8) == 0 )
      goto LABEL_27;
    if ( dword_4F077C4 != 2 )
      goto LABEL_25;
    goto LABEL_204;
  }
LABEL_203:
  if ( (a4[10] & 8) == 0 )
    goto LABEL_27;
LABEL_204:
  if ( (unsigned int)sub_8D23B0(v6) )
    sub_8AE000(v6);
LABEL_25:
  v11[10].m128i_i8[12] |= 8u;
  v13 = dword_4D04820;
  if ( dword_4D04820 )
  {
    v13 = 1;
    sub_658080(v11, 1);
  }
LABEL_27:
  v14 = a4[125];
  if ( (v14 & 2) != 0 )
  {
    v11[10].m128i_i8[15] |= 2u;
  }
  else if ( (v14 & 1) != 0 )
  {
    v11[10].m128i_i8[15] |= 1u;
  }
  else if ( (v14 & 4) != 0 )
  {
    v11[10].m128i_i8[15] |= 4u;
  }
  if ( (a4[561] & 2) != 0 || (a2[1] & 0x180) != 0 && (*(_DWORD *)(*a2 + 176) & 0x44000) == 0 )
  {
    v11[10].m128i_i8[10] |= 0x70u;
    if ( *(_BYTE *)(v10 + 80) != 9 )
      goto LABEL_33;
LABEL_62:
    v25 = sub_880C60();
    *(_QWORD *)(v25 + 24) = v10;
    *(_QWORD *)(v25 + 32) = v10;
    v79 = v25;
    v82 = sub_87E420(*(unsigned __int8 *)(v10 + 80));
    v11[13].m128i_i64[1] = sub_725CE0();
    *(_QWORD *)(v10 + 96) = v79;
    *(_QWORD *)(v79 + 56) = v82;
    if ( (*(_BYTE *)(v5 + 179) & 0x40) != 0 )
      goto LABEL_34;
    goto LABEL_63;
  }
  if ( (unsigned int)sub_867AA0() )
  {
    v11[10].m128i_i8[10] |= 0x70u;
    if ( *(_BYTE *)(v10 + 80) == 9 )
      goto LABEL_62;
  }
LABEL_33:
  if ( (*(_BYTE *)(v5 + 179) & 0x40) != 0 )
  {
LABEL_34:
    v15 = v11[5].m128i_i8[8];
    v11[8].m128i_i8[8] = 2;
    v11[5].m128i_i8[8] = v15 & 0x8F | 0x10;
    goto LABEL_35;
  }
LABEL_63:
  v11[5].m128i_i8[8] = *(_BYTE *)(v5 + 88) & 0x70 | v11[5].m128i_i8[8] & 0x8F;
  if ( (*(_BYTE *)(v5 + 88) & 0x70) == 0x20 )
  {
    v11[8].m128i_i8[8] = (((unsigned __int8)v11[10].m128i_i8[12] >> 5) ^ 1) & 1;
    v13 = a1 + 8;
    sub_649830(v85, a1 + 8, 1);
  }
LABEL_35:
  v16 = (__int64)a4;
  v11[5].m128i_i8[8] = *((_BYTE *)a2 + 12) & 3 | v11[5].m128i_i8[8] & 0xFC;
  sub_648B20(a4);
  v20 = word_4F06418[0];
  if ( word_4F06418[0] == 56 )
  {
    v13 = 0;
    v16 = (unsigned __int16)sub_7BE840(0, 0);
    if ( (unsigned int)sub_692B20(v16) )
      goto LABEL_102;
    v21 = (__int64)&dword_4D04428;
    if ( !dword_4D04428 )
    {
      v20 = word_4F06418[0];
      goto LABEL_38;
    }
    v13 = 0;
    v16 = 0;
    if ( (unsigned __int16)sub_7BE840(0, 0) == 73 )
    {
LABEL_102:
      if ( (*(_BYTE *)(v6 + 140) & 0xFB) == 8 )
      {
        v16 = v6;
        v88 = sub_8D4C10(v6, dword_4F077C4 != 2) & 1;
        v31 = v88;
      }
      else
      {
        v88 = 0;
        v31 = 0;
      }
      v11[10].m128i_i8[14] |= 0x10u;
      v11[11].m128i_i8[0] |= 1u;
      a4[127] |= 4u;
      v32 = *(_QWORD *)&dword_4F063F8;
      *((_QWORD *)a4 + 19) = a4;
      v98[0] = v32;
      if ( word_4F06418[0] == 56 )
      {
        if ( (a4[561] & 2) != 0 )
          goto LABEL_211;
        sub_7B8B50(v16, v31, v21, v17);
        v31 = (unsigned int)v31;
      }
      else
      {
        v11[10].m128i_i8[14] |= 0x40u;
        a4[127] |= 8u;
        a4[176] |= 1u;
      }
      v33 = (unsigned __int8)a4[561];
      if ( (v33 & 2) == 0 )
      {
        if ( (char)a4[124] >= 0 )
          goto LABEL_108;
        v21 = *(unsigned __int8 *)(v6 + 140);
        if ( (_BYTE)v21 == 12 )
        {
          v60 = v6;
          do
          {
            v60 = *(_QWORD *)(v60 + 160);
            v21 = *(unsigned __int8 *)(v60 + 140);
          }
          while ( (_BYTE)v21 == 12 );
        }
        if ( (_BYTE)v21 )
        {
          v31 = 0;
          v16 = (__int64)a4;
          sub_6BDE10(a4, 0);
          v6 = *((_QWORD *)a4 + 36);
          v88 = 0;
          if ( (*(_BYTE *)(v6 + 140) & 0xFB) == 8 )
          {
            v16 = *((_QWORD *)a4 + 36);
            v31 = dword_4F077C4 != 2;
            v88 = sub_8D4C10(v6, v31) & 1;
          }
          v69 = 0;
          v34 = 0;
        }
        else
        {
LABEL_108:
          v80 = dword_4F077BC;
          if ( !dword_4F077BC )
          {
            v69 = 0;
            v34 = 0;
            goto LABEL_110;
          }
          if ( (dword_4F077B4 || qword_4F077A8 <= 0x9CA3u || !(_DWORD)v31) && (v11[10].m128i_i8[12] & 0x20) == 0
            || (a2[1] & 0x180) == 0
            || (*(_DWORD *)(*a2 + 176) & 0x44000) != 0 )
          {
            v69 = 0;
            v34 = 0;
LABEL_219:
            if ( !*(_QWORD *)a4 )
              goto LABEL_214;
            *(_BYTE *)(*(_QWORD *)a4 + 83LL) |= 0x40u;
            v33 = (unsigned __int8)a4[561];
            if ( (v33 & 2) != 0 )
            {
              v81 = (__m128i *)(a4 + 472);
LABEL_117:
              *(_BYTE *)(*(_QWORD *)a4 + 83LL) &= ~0x40u;
              goto LABEL_118;
            }
            v80 = 1;
LABEL_110:
            v81 = (__m128i *)(a4 + 472);
            if ( (v11[10].m128i_i8[12] & 0x20) != 0
              || (v70 = v34,
                  v67 = sub_5F2750(v6, (__int64)v11, v88, (v33 & 2) != 0, *((_BYTE *)a2 + 8) >> 7),
                  v34 = v70,
                  v67) )
            {
              *((_QWORD *)a4 + 67) = v98[0];
              if ( (v11[10].m128i_i8[12] & 0x28) != 0 )
              {
                v35 = (unsigned __int8)a4[124];
                v97 = 0;
                a4[124] = v35 & 0x7F;
                v89 = v35 >> 7;
                if ( dword_4F077C4 == 2 )
                {
                  v71 = v34;
                  v78 = v11[7].m128i_i64[1];
                  v68 = sub_8D23B0(v78);
                  v34 = v71;
                  if ( v68 )
                  {
                    sub_8AE000(v78);
                    v34 = v71;
                  }
                }
                v16 = (__int64)a4;
                v77 = v34;
                v31 = a1 + 8;
                sub_638AC0(a4, a1 + 8, 2, 0, &v97);
                v34 = v77;
                v21 = v89 << 7;
                a4[124] = (v89 << 7) | a4[124] & 0x7F;
              }
              else
              {
                v31 = (__int64)v11;
                v16 = (__int64)a4;
                v91 = v34;
                sub_5F2700((__int64)a4, (__int64)v11, v21, v33, v18, v19);
                v34 = v91;
                *((_QWORD *)a4 + 68) = unk_4F061D8;
              }
              if ( (v11[10].m128i_i8[12] & 0x20) == 0 )
              {
                v16 = (__int64)a4;
                v92 = v34;
                sub_649FB0(a4);
                v34 = v92;
              }
            }
            else
            {
              v31 = v6;
              v16 = (__int64)a4;
              v6 = sub_5F2840((__int64)a4, v6, (__int64)v98);
              sub_6BBA30(a4);
              v34 = v70;
            }
            if ( !v80 )
              goto LABEL_118;
            goto LABEL_117;
          }
          v61 = *(_DWORD **)(v10 + 104);
          if ( !v61 )
            v61 = (_DWORD *)sub_8790A0(v10, v31);
          *v61 = dword_4F06650[0];
          v16 = sub_63B4E0(v10);
          v69 = v16;
          sub_7BC160(v16);
          v34 = 1;
        }
LABEL_213:
        if ( !dword_4F077BC )
        {
LABEL_214:
          v33 = (unsigned __int8)a4[561];
          if ( (a4[561] & 2) != 0 )
          {
            v81 = (__m128i *)(a4 + 472);
LABEL_118:
            if ( v34 && word_4F06418[0] == 9 )
              sub_7B8B50(v16, v31, v21, v33);
            v22 = 0;
            goto LABEL_122;
          }
          v80 = 0;
          goto LABEL_110;
        }
        goto LABEL_219;
      }
LABEL_211:
      if ( v84[2].m128i_i64[0] )
      {
        v16 = v10;
        v55 = sub_63B4E0(v10);
        v34 = 0;
        v69 = v55;
      }
      else
      {
        v66 = (const __m128i *)sub_892880(v10, v31);
        *v84 = _mm_loadu_si128(v66);
        v84[1] = _mm_loadu_si128(v66 + 1);
        v84[2].m128i_i64[0] = v66[2].m128i_i64[0];
        v16 = (__int64)v84;
        v31 = 1;
        v69 = sub_63B4E0(v10);
        sub_879020(v84, 1);
        v34 = 0;
      }
      goto LABEL_213;
    }
    v20 = word_4F06418[0];
  }
  v21 = dword_4D04428;
  if ( dword_4D04428 && v20 == 73 )
    goto LABEL_102;
LABEL_38:
  v81 = (__m128i *)(a4 + 472);
  if ( v20 == 19 )
  {
    a4[127] |= 4u;
    v51 = v11[7].m128i_i64[1];
    *(__int32 *)((char *)&v11[10].m128i_i32[3] + 2) = *(__int32 *)((_BYTE *)&v11[10].m128i_i32[3] + 2) & 0xFDFFEF
                                                    | 0x6020010;
    v52 = sub_8250F0(v51, &unk_4F04DA0);
    *(_QWORD *)(v52 + 8) = v11;
    v11[11].m128i_i64[1] = v52;
    sub_7B8B50(v51, &unk_4F04DA0, v53, v54);
    goto LABEL_45;
  }
  v22 = (__int64)&dword_4F077BC;
  if ( !dword_4F077BC )
  {
    if ( (a4[561] & 2) == 0 )
      goto LABEL_41;
    v87 = *((_BYTE *)a2 + 8);
    v22 = v87;
    v21 = v87 & 0xC0;
    if ( (v87 & 0xC0) != 0x40 )
      goto LABEL_41;
  }
  if ( v20 != 16 )
  {
    if ( v20 == 56 )
    {
      v13 = 0;
      v16 = 0;
      if ( (unsigned __int16)sub_7BE840(0, 0) == 16 )
      {
        v28 = word_4F06418[0];
        a4[127] |= 4u;
        v11[10].m128i_i8[14] |= 0x10u;
        v11[11].m128i_i8[0] |= 1u;
        if ( v28 == 56 )
          sub_7B8B50(0, 0, v21, v22);
        goto LABEL_225;
      }
    }
LABEL_41:
    v23 = v11[10].m128i_i8[12];
    if ( (v23 & 8) != 0 && (dword_4F077C4 != 2 || unk_4F07778 <= 201702) )
    {
      sub_6851C0(2429, &dword_4F063F8);
      v11[11].m128i_i8[1] = 1;
      v11[11].m128i_i64[1] = sub_72C9A0();
LABEL_45:
      v22 = 0;
      goto LABEL_46;
    }
    if ( (v23 & 0x20) == 0 )
    {
      v86 = 1;
      v69 = 0;
      goto LABEL_127;
    }
    if ( (a4[10] & 0x20) != 0 )
      v11[10].m128i_i8[12] |= 0x10u;
    if ( (unsigned int)sub_8D32E0(v11[7].m128i_i64[1]) )
    {
      sub_6854E0(252, v10);
    }
    else if ( (unsigned int)sub_63BB10(*(_QWORD *)a4, &dword_4F063F8) )
    {
      sub_649FB0(a4);
      *(_BYTE *)(*(_QWORD *)a4 + 84LL) |= 0x80u;
    }
    v29 = v11[7].m128i_i64[1];
    if ( (*(_BYTE *)(v29 + 140) & 0xFB) != 8
      || (sub_8D4C10(v29, dword_4F077C4 != 2) & 1) == 0
      || (unsigned int)sub_8D5A50(v11[7].m128i_i64[1]) )
    {
      goto LABEL_45;
    }
    if ( dword_4F077BC && ((v11[10].m128i_i8[12] & 8) == 0 || *(_BYTE *)(v10 + 80) == 21) )
      v30 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 ? 8 : 5;
    else
      v30 = 8;
    sub_685440(v30, 257, v10);
    v22 = 0;
    v69 = 0;
LABEL_122:
    v18 = dword_4D04820;
    if ( !dword_4D04820 )
      goto LABEL_47;
    goto LABEL_123;
  }
  a4[127] |= 4u;
  v11[10].m128i_i8[14] |= 0x10u;
  v11[11].m128i_i8[0] |= 1u;
LABEL_225:
  if ( (a4[561] & 2) == 0 )
  {
    v16 = v10;
    v13 = a2[4];
    sub_892A30(v10, v13);
    v64 = *(_QWORD *)(v10 + 104);
    if ( v64 )
    {
      v65 = *(_QWORD *)(v64 + 8);
      if ( v65 )
      {
        if ( *(_WORD *)(*(_QWORD *)(v65 + 8) + 24LL) == 73 )
        {
          v11[10].m128i_i8[14] |= 0x40u;
          a4[127] |= 8u;
          a4[176] |= 1u;
        }
      }
    }
  }
  sub_7B8B50(v16, v13, v21, v22);
  v11[8].m128i_i8[8] = 1;
  v22 = 1;
LABEL_46:
  v69 = 0;
  v18 = dword_4D04820;
  if ( !dword_4D04820 )
  {
LABEL_47:
    v86 = 1;
    v23 = v11[10].m128i_i8[12];
    goto LABEL_127;
  }
LABEL_123:
  v23 = v11[10].m128i_i8[12];
  v86 = 1;
  if ( (v23 & 0x20) != 0 )
  {
    if ( (a4[561] & 2) != 0 || (v22 &= 1u, (_DWORD)v22) )
    {
      v86 = 3;
    }
    else
    {
      v58 = v11[7].m128i_i64[1];
      if ( dword_4F077C4 == 2 )
      {
        if ( (unsigned int)sub_8D23B0(v58) )
          sub_8AE000(v58);
        v58 = v11[7].m128i_i64[1];
      }
      if ( !(unsigned int)sub_8D23B0(v58) || (unsigned int)sub_8DBE70(v11[7].m128i_i64[1]) )
      {
        v86 = 3;
        v23 = v11[10].m128i_i8[12];
      }
      else
      {
        v90 = v11[7].m128i_i64[1];
        v63 = sub_67F240(v90);
        sub_685A50(v63, v73, v90, 8);
        v86 = 3;
        v23 = v11[10].m128i_i8[12];
      }
    }
  }
LABEL_127:
  sub_644920(a4, (v23 & 0x20) != 0, v21, v22, v18, v19);
  if ( a3 )
  {
    v74 = *(_QWORD *)(a3 + 336);
    if ( *(_QWORD *)(v74 + 96) )
    {
      sub_86A080();
      *(_QWORD *)(v74 + 96) = 0;
    }
  }
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    v36 = v86;
    BYTE1(v36) = BYTE1(v86) | 8;
    if ( (v11[11].m128i_i8[0] & 1) == 0 )
      v36 = v86;
    LOWORD(v86) = v36;
    sub_8756F0(v36, v10, a1 + 8, *((_QWORD *)a4 + 44));
  }
  v37 = v81;
  sub_729470(v11, v81);
  if ( (v11[10].m128i_i8[12] & 4) != 0 )
    v11[12] = _mm_loadu_si128((const __m128i *)(a4 + 536));
  if ( dword_4F04C64 != -1
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 2) != 0
    && dword_4F077C4 == 2
    && (v11[-1].m128i_i8[8] & 1) != 0
    && (*(_BYTE *)(a1 + 18) & 0x40) == 0
    && (v37 = v11, v50 = sub_7CAFF0(a1, v11, qword_4F04C68), (v38 = v50) != 0) )
  {
    if ( (v11[11].m128i_i8[0] & 1) != 0 )
      *(_BYTE *)(v50 + 33) |= 0x10u;
  }
  else
  {
    v38 = 0;
  }
  if ( !dword_4F04C3C && (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    v39 = v75;
    if ( !v75 )
      v39 = v6;
    v76 = v39;
    if ( *(_BYTE *)(v10 + 80) == 21 )
    {
      if ( (v86 & 2) == 0 )
      {
        v62 = sub_8921F0(v84[6].m128i_i64[1], v37, v38);
        *(_BYTE *)(v62 + 57) |= 0x20u;
        *(_QWORD *)(v62 + 32) = v76;
LABEL_147:
        sub_65C210(a4);
        if ( (a4[127] & 4) != 0 )
          sub_6522D0(a4);
        goto LABEL_149;
      }
    }
    else if ( (v86 & 0x802) == 0 )
    {
      sub_86A3D0(v11, v39, v38, (a4[126] & 4) == 0 ? 16 : 80, v81);
      goto LABEL_147;
    }
    v11[16].m128i_i64[0] = v39;
    v11[8].m128i_i8[9] = a4[268];
    goto LABEL_147;
  }
LABEL_149:
  sub_854980(v10, 0);
  v40 = a2[4];
  if ( (v40 || (a2[1] & 0x180) != 0 || (a4[561] & 2) != 0)
    && (*(_BYTE *)(v5 + 178) & 4) == 0
    && (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    v41 = a4[561] & 2;
    if ( (a2[1] & 0x180) != 0 || v41 && (a2[1] & 0xC0) != 0x40 )
    {
      if ( *(_BYTE *)(v10 + 80) == 9 )
        v42 = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 56LL);
      else
        v42 = *(_QWORD *)(v10 + 88);
      if ( v41 )
      {
        v43 = *(_QWORD *)(a3 + 336);
      }
      else
      {
        v83 = v42;
        v59 = sub_727340();
        *(_BYTE *)(v59 + 120) = 5;
        v94 = v59;
        sub_877D80(v59, v10);
        *(_BYTE *)(v94 + 88) = *(_BYTE *)(v94 + 88) & 0x8F | 0x20;
        sub_877E20(0, v94, v5);
        sub_7344C0(v94, unk_4F04C5C);
        v42 = v83;
        v43 = v94;
      }
      *(_QWORD *)(v11[13].m128i_i64[1] + 16) = v43;
      *(_QWORD *)(v42 + 192) = v11;
      v44 = 0;
      *(_BYTE *)(v43 + 88) = v11[5].m128i_i8[8] & 3 | *(_BYTE *)(v43 + 88) & 0xFC;
      *(_QWORD *)(v42 + 104) = v43;
      if ( *(char *)(v5 + 177) < 0 )
        v44 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v5 + 168) + 160LL) + 121LL) & 1;
      *(_BYTE *)(v43 + 121) = v44 | *(_BYTE *)(v43 + 121) & 0xFE;
      if ( dword_4F07590 )
        *(_QWORD *)(v43 + 192) = v11;
      *(_QWORD *)(v43 + 200) = v43;
      if ( (v86 & 2) != 0 )
        *(_QWORD *)(v43 + 208) = v43;
      *(_DWORD *)(v42 + 64) = v72;
    }
    else
    {
      sub_896480(v10, v40, v72);
    }
    if ( (a4[561] & 2) != 0 )
    {
      sub_879080(v84, v69, *(_QWORD *)(a3 + 192));
      if ( (v86 & 2) != 0 )
        *(_DWORD *)(a3 + 36) = 1;
    }
  }
  if ( dword_4F077BC )
  {
    v45 = v11[10].m128i_i8[8];
    if ( (v45 & 7) == 0 )
      v11[10].m128i_i8[8] = *(_BYTE *)(*(_QWORD *)(v5 + 168) + 109LL) & 7 | v45 & 0xF8;
    v46 = *((_QWORD *)a4 + 30);
    if ( v46 )
      v11[9].m128i_i64[0] = v46;
  }
  if ( v11[6].m128i_i64[1] )
  {
    v47 = 0;
    if ( (a4[122] & 1) == 0 )
      v47 = (*(_BYTE *)(*(_QWORD *)a4 + 81LL) & 2) != 0;
    sub_656C00(a4, 7, v11, v47, a4[122] & 1);
  }
  if ( v11[5].m128i_i8[10] >= 0 )
    sub_8D9350(v6, a1 + 8);
  if ( qword_4CF8008 && *(_QWORD *)(qword_4CF8008 + 128) )
    *(_QWORD *)(qword_4CF8008 + 16) = v10;
  sub_8D9610(v6, v98);
  v48 = *(_WORD *)(v5 + 180) & 0xFC3F;
  result = v48 | ((unsigned __int8)(v98[0] & 0xF | (*(_WORD *)(v5 + 180) >> 6) & 0xF) << 6);
  *(_WORD *)(v5 + 180) = v48 | ((v98[0] & 0xF | (*(_WORD *)(v5 + 180) >> 6) & 0xF) << 6);
  return result;
}
