// Function: sub_743600
// Address: 0x743600
//
const __m128i *__fastcall sub_743600(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        _QWORD *a4,
        _DWORD *a5,
        unsigned int a6,
        int *a7,
        __m128i *a8,
        __m128i *a9)
{
  int v11; // ebx
  __int8 v12; // al
  int v13; // eax
  __int64 v15; // rdi
  __m128i *v16; // r15
  __int64 v17; // rbx
  __int8 v19; // al
  __m128i *v20; // r12
  int *v21; // rax
  int v22; // eax
  int v23; // esi
  __int64 v24; // r15
  __m128i *v25; // r15
  __m128i *v26; // rax
  bool v27; // zf
  int v28; // r9d
  __m128i *v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r14
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r15
  __int64 v42; // rax
  const __m128i *v43; // rax
  __m128i *v44; // rbx
  const __m128i *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rdx
  char v48; // al
  __m128i *v49; // r12
  unsigned __int64 v50; // r15
  __int64 i; // rbx
  __int8 v52; // al
  __int8 v53; // al
  unsigned __int64 v54; // rsi
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // r15
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int32 v63; // eax
  const __m128i *v64; // rdi
  _QWORD *v65; // rbx
  __int64 v66; // r13
  __int64 v67; // r14
  __int32 v68; // eax
  __int64 v69; // rdi
  unsigned __int8 v70; // r12
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  int v75; // eax
  _DWORD *v76; // rax
  __int64 v77; // rdx
  _DWORD *v78; // rax
  _DWORD *v79; // r9
  _DWORD *v80; // r8
  _DWORD *v81; // rdx
  _DWORD *v82; // rdi
  const __m128i *v83; // rdi
  __int64 v84; // [rsp+8h] [rbp-A8h]
  __int64 v85; // [rsp+8h] [rbp-A8h]
  int v86; // [rsp+10h] [rbp-A0h]
  int v88; // [rsp+10h] [rbp-A0h]
  int v90; // [rsp+10h] [rbp-A0h]
  _DWORD *v92; // [rsp+10h] [rbp-A0h]
  _DWORD *v93; // [rsp+10h] [rbp-A0h]
  __m128i *v94; // [rsp+18h] [rbp-98h] BYREF
  int v95; // [rsp+28h] [rbp-88h] BYREF
  int v96; // [rsp+2Ch] [rbp-84h] BYREF
  const __m128i *v97; // [rsp+30h] [rbp-80h] BYREF
  __m128i *v98; // [rsp+38h] [rbp-78h] BYREF
  __int64 v99[5]; // [rsp+40h] [rbp-70h] BYREF
  int v100; // [rsp+68h] [rbp-48h]
  __m128i *v101; // [rsp+70h] [rbp-40h]
  char v102; // [rsp+78h] [rbp-38h]

  v11 = a6;
  v97 = a1;
  v12 = a1[10].m128i_i8[13];
  v94 = a2;
  v95 = 0;
  if ( v12 == 12 )
  {
    switch ( a1[11].m128i_i8[0] )
    {
      case 0:
        v40 = sub_8A4460(&a1[11].m128i_u64[1], a6, &v94, a3);
        v41 = v40;
        if ( !v40 )
          goto LABEL_113;
        if ( *(_BYTE *)(v40 + 8) != 1 || (*(_BYTE *)(v40 + 24) & 1) != 0 )
          goto LABEL_59;
        v64 = *(const __m128i **)(v40 + 32);
        if ( !v64 )
        {
LABEL_113:
          if ( (v11 & 0x2000) == 0 )
            goto LABEL_25;
          v63 = a8[5].m128i_i32[2];
          if ( a1[10].m128i_i8[13] == 12 && (a1[11].m128i_i8[1] & 4) != 0 )
            v63 |= 1u;
          a8[5].m128i_i32[2] = v63;
          v13 = *a7;
          goto LABEL_7;
        }
        v27 = v64[9].m128i_i64[0] == 0;
        v97 = *(const __m128i **)(v40 + 32);
        if ( !v27 && (v11 & 0x86150) == 0x10 )
        {
          sub_72A510(v64, a9);
          v97 = 0;
          a9[9].m128i_i64[0] = 0;
        }
        v65 = (_QWORD *)a8[4].m128i_i64[0];
        if ( !v65 )
          goto LABEL_129;
        v66 = a1[11].m128i_u32[2];
        v67 = v65[2];
        v68 = a1[11].m128i_i32[2];
        if ( v66 < v67 )
          v66 = v65[2];
        if ( v66 > v67 )
        {
          if ( v66 > v65[1] )
          {
            v85 = v65[1];
            v92 = (_DWORD *)*v65;
            v78 = (_DWORD *)sub_823970(4 * v66);
            v79 = v92;
            v80 = v78;
            if ( v67 > 0 )
            {
              v81 = v92;
              v82 = &v78[v67];
              do
              {
                if ( v78 )
                  *v78 = *v81;
                ++v78;
                ++v81;
              }
              while ( v82 != v78 );
            }
            v93 = v80;
            sub_823A00(v79, 4 * v85);
            v65[1] = v66;
            *v65 = v93;
          }
          v76 = (_DWORD *)(*v65 + 4 * v67);
          v77 = *v65 + 4 * v66;
          do
          {
            if ( v76 )
              *v76 = 0;
            ++v76;
            ++v65[2];
          }
          while ( v76 != (_DWORD *)v77 );
        }
        else
        {
          if ( v66 >= v67 )
            goto LABEL_128;
          v65[2] = v66;
        }
        v68 = a1[11].m128i_i32[2];
LABEL_128:
        *(_DWORD *)(*v65 + 4LL * (unsigned int)(v68 - 1)) = 1;
LABEL_129:
        if ( (*(_BYTE *)(v41 + 24) & 0x10) == 0 )
          goto LABEL_25;
        a8[5].m128i_i32[2] = 1;
        v13 = *a7;
        goto LABEL_7;
      case 1:
        if ( (unsigned int)sub_72E9D0(a1, &v98, &v96) )
        {
          v97 = (const __m128i *)sub_744640(
                                   (_DWORD)a1,
                                   (_DWORD)v98,
                                   v96,
                                   (_DWORD)v94,
                                   a3,
                                   (_DWORD)a5,
                                   v11,
                                   (__int64)a7,
                                   (__int64)a8,
                                   (__int64)a9);
          v13 = *a7;
          goto LABEL_7;
        }
        v49 = (__m128i *)sub_72E9A0((__int64)a1);
        if ( v49->m128i_i64[0] == dword_4D03B80
          && (v49[1].m128i_i8[11] & 2) != 0
          && v49[1].m128i_i8[8] == 1
          && v49[3].m128i_i8[8] == 5 )
        {
          v49 = (__m128i *)v49[4].m128i_i64[1];
        }
        v50 = sub_7410C0(v49, v94, a3, a4, a5, v11, a7, a8->m128i_i64, a9, &v97);
        v21 = a7;
        if ( !v50 )
          goto LABEL_26;
        if ( *a7 )
          goto LABEL_79;
        v99[0] = 0;
        v99[1] = 0;
        if ( (unsigned int)sub_7A30C0(v50, 0, 0, a9) )
        {
          sub_67E3D0(v99);
          v13 = *a7;
          goto LABEL_7;
        }
        sub_67E3D0(v99);
LABEL_79:
        if ( (unk_4F07734 || (*(_BYTE *)(v50 + 25) & 1) != 0) && (v11 & 0x10000) == 0 )
        {
LABEL_59:
          *a7 = 1;
          return (const __m128i *)sub_72C9A0();
        }
        if ( v49 != (__m128i *)v50 )
        {
          sub_70FD90((__int64 *)v50, (__int64)a9);
          v97 = 0;
          v13 = *a7;
          goto LABEL_7;
        }
        goto LABEL_25;
      case 2:
        v43 = (const __m128i *)sub_740BE0(
                                 (__int64)a1,
                                 (int)v94,
                                 a3,
                                 (__int64)a4,
                                 0,
                                 0,
                                 0,
                                 (__int64)a5,
                                 a8,
                                 a6,
                                 a7,
                                 (__int64)a9);
        goto LABEL_61;
      case 3:
        v43 = (const __m128i *)sub_740BE0(
                                 (__int64)a1,
                                 (int)v94,
                                 a3,
                                 (__int64)a4,
                                 1,
                                 0,
                                 0,
                                 (__int64)a5,
                                 a8,
                                 a6,
                                 a7,
                                 (__int64)a9);
        goto LABEL_61;
      case 4:
        v43 = (const __m128i *)sub_740BE0(
                                 a1[11].m128i_i64[1],
                                 (int)v94,
                                 a3,
                                 (__int64)a4,
                                 1,
                                 0,
                                 0,
                                 (__int64)a5,
                                 a8,
                                 a6,
                                 a7,
                                 (__int64)a9);
        goto LABEL_61;
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 0xA:
        v25 = (__m128i *)sub_72F1F0((__int64)a1);
        if ( v25 )
        {
          v26 = (__m128i *)sub_724DC0();
          v27 = a1[11].m128i_i8[0] == 10;
          v28 = 4;
          v98 = v26;
          if ( v27 )
            v28 = 2052;
          v29 = v94;
          v25 = (__m128i *)sub_7410C0(v25, v94, a3, 0, a5, v11 | (unsigned int)v28, a7, a8->m128i_i64, v26, v99);
          if ( !v25 )
          {
            if ( v99[0] )
              v25 = (__m128i *)sub_730690(v99[0]);
            else
              v25 = (__m128i *)sub_73A720(v98, (__int64)v94);
          }
          v30 = v25->m128i_i64[0];
          sub_724E30((__int64)&v98);
        }
        else
        {
          v29 = v94;
          v30 = sub_8A2270(a1[11].m128i_i64[1], (_DWORD)v94, a3, (_DWORD)a5, v11, (_DWORD)a7, (__int64)a8);
          if ( (unsigned int)sub_8D32E0(v30) )
            v30 = sub_8D46C0(v30);
        }
        v31 = a1[11].m128i_i64[1];
        if ( v31 == v30
          || v30 && v31 && dword_4F07588 && (v32 = *(_QWORD *)(v30 + 32), *(_QWORD *)(v31 + 32) == v32) && v32 )
        {
          if ( v25 == (__m128i *)sub_72F1F0((__int64)a1) )
            goto LABEL_25;
        }
        if ( (unsigned int)sub_8DBE70(v30) )
        {
          *a9 = _mm_loadu_si128(a1);
          a9[1] = _mm_loadu_si128(a1 + 1);
          a9[2] = _mm_loadu_si128(a1 + 2);
          a9[3] = _mm_loadu_si128(a1 + 3);
          a9[4] = _mm_loadu_si128(a1 + 4);
          a9[5] = _mm_loadu_si128(a1 + 5);
          a9[6] = _mm_loadu_si128(a1 + 6);
          a9[7] = _mm_loadu_si128(a1 + 7);
          a9[8] = _mm_loadu_si128(a1 + 8);
          a9[9] = _mm_loadu_si128(a1 + 9);
          a9[10] = _mm_loadu_si128(a1 + 10);
          a9[11] = _mm_loadu_si128(a1 + 11);
          a9[12] = _mm_loadu_si128(a1 + 12);
          if ( a1[11].m128i_i64[1] )
            a9[11].m128i_i64[1] = v30;
          a9[12].m128i_i64[0] = (__int64)v25;
          v97 = 0;
          v13 = *a7;
          goto LABEL_7;
        }
        for ( i = v30; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
          sub_8AE000(i);
        v52 = a1[11].m128i_i8[0];
        if ( v52 == 9 )
        {
          sub_73C780(i, 0, (__int64)a9);
          v13 = *a7;
          goto LABEL_55;
        }
        if ( v52 == 10 )
        {
          v70 = *(_BYTE *)(sub_72C390() + 160);
          v75 = sub_731B40((__int64)v25, v29, v71, v72, v73, v74);
          sub_72BBE0((__int64)a9, v75 == 0, v70);
          a9[8].m128i_i64[0] = sub_72C390();
          v13 = *a7;
          goto LABEL_55;
        }
        if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
          sub_8AE000(i);
        if ( (unsigned int)sub_8D23B0(i) || (unsigned int)sub_8D2310(i) )
          goto LABEL_21;
        v53 = a1[11].m128i_i8[0];
        switch ( v53 )
        {
          case 5:
            v54 = *(_QWORD *)(i + 128);
            break;
          case 6:
            v54 = sub_8D4B80(i);
            break;
          case 7:
            v54 = (unsigned int)sub_731F80(v30, v25 == 0, (__int64)v25, 0, a7, (int *)v99);
            break;
          default:
            goto LABEL_75;
        }
        sub_72BBE0((__int64)a9, v54, byte_4F06A51[0]);
        v13 = *a7;
        goto LABEL_55;
      case 0xB:
        v84 = a1[11].m128i_i64[1];
        v90 = a3;
        v42 = sub_690FF0(0, a1[12].m128i_i64[0], 0, (int)v94, a3, (__int64)a5, a6, (__int64)a7, (__int64)a8);
        v43 = (const __m128i *)sub_740BE0(
                                 v84,
                                 (int)v94,
                                 v90,
                                 (__int64)a4,
                                 1,
                                 1u,
                                 v42,
                                 (__int64)a5,
                                 a8,
                                 v11,
                                 a7,
                                 (__int64)a9);
LABEL_61:
        v97 = v43;
        v13 = *a7;
        goto LABEL_7;
      case 0xC:
        v44 = (__m128i *)sub_743600(
                           a1[11].m128i_i64[1],
                           (_DWORD)v94,
                           a3,
                           (_DWORD)a4,
                           (_DWORD)a5,
                           a6,
                           (__int64)a7,
                           (__int64)a8,
                           (__int64)a9);
        if ( *a7 )
          return (const __m128i *)sub_72C9A0();
        if ( !v44 )
          v44 = sub_740630(a9);
        v97 = (const __m128i *)sub_724D80(12);
        sub_7249B0((__int64)v97, 12);
        v45 = v97;
        v46 = v44[8].m128i_i64[0];
        v97[11].m128i_i64[1] = (__int64)v44;
        v45[8].m128i_i64[0] = v46;
        v13 = *a7;
        goto LABEL_7;
      default:
        goto LABEL_75;
    }
  }
  if ( v12 != 6 )
  {
    if ( v12 != 15 )
      goto LABEL_4;
    sub_72A510(a1, a9);
    v19 = a9[11].m128i_i8[0];
    v20 = v94;
    if ( v19 != 48 )
    {
      if ( v19 != 6 )
      {
        if ( v19 == 13 )
        {
          v33 = (__int64 *)a9[11].m128i_i64[1];
          v98 = (__m128i *)sub_724DC0();
          sub_6E3D60((__int64)v99);
          v100 = v11;
          v99[3] = (__int64)v94;
          v99[4] = a3;
          v101 = a8;
          v34 = sub_6DD8E0(v33, 0, (__int64)v99, v98);
          v39 = v34;
          if ( v102 )
          {
            *a7 = 1;
          }
          else if ( v34 )
          {
            if ( !(unsigned int)sub_731EE0(v34, 0, v35, v36, v37, v38) )
              a9[8].m128i_i64[0] = sub_72CD60();
            a9[11].m128i_i64[1] = v39;
          }
          else
          {
            if ( !(unsigned int)sub_7322D0((__int64)v98, 0, v35, v36, v37, v38) )
              a9[8].m128i_i64[0] = sub_72CD60();
            a9[11].m128i_i64[1] = sub_724E50((__int64 *)&v98, 0);
          }
          if ( v98 )
            sub_724E30((__int64)&v98);
          goto LABEL_54;
        }
        if ( v19 != 2 )
        {
LABEL_21:
          v97 = 0;
          *a7 = 1;
          return (const __m128i *)sub_72C9A0();
        }
        v57 = a9[11].m128i_i64[1];
LABEL_100:
        v99[0] = (__int64)sub_724DC0();
        v58 = sub_743600(v57, (_DWORD)v20, a3, 0, (_DWORD)a5, v11, (__int64)a7, (__int64)a8, v99[0]);
        if ( !a7 )
        {
          if ( !v58 )
            v58 = sub_724E50(v99, v20);
          a9[11].m128i_i64[1] = v58;
          if ( !(unsigned int)sub_7322D0(v58, (__int64)v20, v59, v60, v61, v62) )
            a9[8].m128i_i64[0] = sub_72CD60();
        }
        if ( v99[0] )
          sub_724E30((__int64)v99);
LABEL_54:
        v13 = *a7;
LABEL_55:
        v97 = 0;
        goto LABEL_7;
      }
      v55 = a9[11].m128i_i64[1];
LABEL_97:
      v56 = sub_8A2270(v55, (_DWORD)v20, a3, (_DWORD)a5, v11, (_DWORD)a7, (__int64)a8);
      a9[11].m128i_i64[1] = v56;
      if ( !(unsigned int)sub_8DC060(v56) )
      {
        a9[8].m128i_i64[0] = sub_72CD60();
        v13 = *a7;
        goto LABEL_55;
      }
      goto LABEL_54;
    }
    v47 = a9[11].m128i_i64[1];
    v48 = *(_BYTE *)(v47 + 8);
    switch ( v48 )
    {
      case 1:
        a9[11].m128i_i8[0] = 2;
        v57 = *(_QWORD *)(v47 + 32);
        a9[11].m128i_i64[1] = v57;
        goto LABEL_100;
      case 2:
        a9[11].m128i_i8[0] = 59;
        a9[11].m128i_i64[1] = *(_QWORD *)(v47 + 32);
        goto LABEL_21;
      case 0:
        a9[11].m128i_i8[0] = 6;
        v55 = *(_QWORD *)(v47 + 32);
        a9[11].m128i_i64[1] = v55;
        goto LABEL_97;
    }
LABEL_75:
    sub_721090();
  }
  if ( a1[11].m128i_i8[0] == 3 )
  {
    v15 = a1[11].m128i_i64[1];
    if ( *(_BYTE *)(v15 + 173) == 12 )
    {
      v86 = a3;
      v16 = (__m128i *)sub_743600(v15, (_DWORD)v94, a3, 0, (_DWORD)a5, a6, (__int64)a7, (__int64)a8, (__int64)a9);
      if ( *a7 )
        return (const __m128i *)sub_72C9A0();
      v17 = sub_8A2270(a1[8].m128i_i64[0], (_DWORD)v94, v86, (_DWORD)a5, v11, (_DWORD)a7, (__int64)a8);
      if ( *a7 )
        return (const __m128i *)sub_72C9A0();
      if ( !v16 )
        v16 = sub_740630(a9);
      v97 = 0;
      sub_72A510(a1, a9);
      a9[8].m128i_i64[0] = v17;
      a9[11].m128i_i64[1] = (__int64)v16;
      v13 = *a7;
      goto LABEL_7;
    }
  }
LABEL_4:
  if ( unk_4F07734 && (v88 = a3, (unsigned int)sub_8DC060(a1[8].m128i_i64[0])) )
  {
    v22 = v11;
    v23 = v11 & 0x80;
    if ( (v11 & 0x80) != 0 )
    {
      LOBYTE(v22) = v11 & 0x7F;
      v11 = v22;
    }
    v24 = sub_8A2270(a1[8].m128i_i64[0], (_DWORD)v94, v88, (_DWORD)a5, v11, (_DWORD)a7, (__int64)a8);
    if ( *a7 )
      return (const __m128i *)sub_72C9A0();
    if ( (a1[10].m128i_i8[8] & 0x20) == 0
      || a1[10].m128i_i8[13] != 10
      || (unsigned int)sub_8D3BB0(v24)
      || (unsigned int)sub_8DC060(v24) )
    {
      sub_72A510(a1, a9);
      a9[10].m128i_i8[11] &= ~4u;
      a9[8].m128i_i64[0] = v24;
      a9[9].m128i_i64[0] = 0;
    }
    else
    {
      v69 = a1[11].m128i_i64[0];
      if ( v69 )
      {
        if ( !*(_QWORD *)(v69 + 120)
          && (unsigned int)sub_728A90(v69, v24, (a1[10].m128i_i8[8] & 0x20) != 0, v23 != 0, &v95) )
        {
          BYTE1(v11) |= 0x90u;
          v83 = (const __m128i *)sub_743600(
                                   v69,
                                   (_DWORD)v94,
                                   v88,
                                   v24,
                                   (_DWORD)a5,
                                   v11,
                                   (__int64)a7,
                                   (__int64)a8,
                                   (__int64)a9);
          if ( v83 )
            sub_72A510(v83, a9);
          sub_7115B0(a9, v24, 0, 1, 1, 1, 0, 0, 1u, v95, 0, &v98, v99, a5);
          if ( !(LODWORD(v99[0]) | (unsigned int)v98) )
            goto LABEL_34;
        }
      }
      else if ( (unsigned int)sub_72FDF0(v24, a9) )
      {
        goto LABEL_34;
      }
      *a7 = 1;
    }
  }
  else
  {
    if ( (a1[10].m128i_i8[11] & 4) != 0 )
    {
      sub_72A510(a1, a9);
      a9[10].m128i_i8[11] &= ~4u;
      v97 = 0;
      a9[9].m128i_i64[0] = 0;
      v13 = *a7;
      goto LABEL_7;
    }
    if ( !v97[9].m128i_i64[0] || (v11 & 0x86150) != 0x10 )
      goto LABEL_25;
    sub_72A510(v97, a9);
    a9[9].m128i_i64[0] = 0;
  }
LABEL_34:
  v97 = 0;
LABEL_25:
  v21 = a7;
LABEL_26:
  v13 = *v21;
LABEL_7:
  if ( !v13 )
    return v97;
  return (const __m128i *)sub_72C9A0();
}
