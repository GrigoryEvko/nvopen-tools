// Function: sub_36E0E00
// Address: 0x36e0e00
//
__int64 __fastcall sub_36E0E00(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // r13
  __int64 v11; // rax
  __int16 v12; // cx
  __int64 v13; // rax
  unsigned int v14; // r15d
  __int64 v15; // r9
  bool v16; // al
  __int64 v17; // rax
  __int64 v18; // rbx
  unsigned __int64 v19; // rdx
  __int64 v20; // r12
  int v21; // r14d
  __int16 v22; // r13
  unsigned __int16 *v23; // rax
  unsigned __int16 *v24; // rax
  unsigned __int16 *v25; // rsi
  _QWORD *v26; // rdi
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r10
  int v30; // r14d
  __m128i v31; // xmm0
  int v32; // eax
  __m128i v33; // xmm1
  unsigned __int64 v34; // rax
  __int64 v35; // r9
  unsigned __int64 v36; // r13
  __int64 v38; // rsi
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r14
  __int16 v44; // si
  __int64 v45; // rdx
  unsigned __int16 *v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned int v50; // r15d
  __int64 v51; // r14
  __int128 v52; // rax
  __int64 v53; // r9
  char v54; // al
  __int128 v55; // [rsp-30h] [rbp-1B0h]
  __int16 v56; // [rsp+Ah] [rbp-176h]
  __int64 v57; // [rsp+10h] [rbp-170h]
  __int64 v58; // [rsp+20h] [rbp-160h]
  __int64 v59; // [rsp+20h] [rbp-160h]
  unsigned int v60; // [rsp+20h] [rbp-160h]
  unsigned int v61; // [rsp+28h] [rbp-158h]
  unsigned __int64 v62; // [rsp+28h] [rbp-158h]
  int v63; // [rsp+28h] [rbp-158h]
  __int64 v64; // [rsp+30h] [rbp-150h]
  __int64 v65; // [rsp+30h] [rbp-150h]
  __int64 v66; // [rsp+30h] [rbp-150h]
  unsigned __int64 v67; // [rsp+38h] [rbp-148h]
  unsigned int v68; // [rsp+38h] [rbp-148h]
  __int64 v69; // [rsp+38h] [rbp-148h]
  unsigned __int16 v70; // [rsp+48h] [rbp-138h]
  _QWORD *v71; // [rsp+48h] [rbp-138h]
  __int64 v72; // [rsp+50h] [rbp-130h]
  unsigned __int64 v73; // [rsp+58h] [rbp-128h]
  __int64 v74; // [rsp+68h] [rbp-118h]
  __m128i v75; // [rsp+70h] [rbp-110h] BYREF
  __m128i v76; // [rsp+80h] [rbp-100h] BYREF
  __m128i v77; // [rsp+90h] [rbp-F0h] BYREF
  __m128i v78; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v79; // [rsp+B0h] [rbp-D0h] BYREF
  int v80; // [rsp+B8h] [rbp-C8h]
  _OWORD v81[2]; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+E0h] [rbp-A0h]
  int v83; // [rsp+E8h] [rbp-98h]
  unsigned __int16 *v84; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v85; // [rsp+F8h] [rbp-88h]
  _BYTE v86[128]; // [rsp+100h] [rbp-80h] BYREF

  v4 = 80;
  v5 = a2;
  v6 = a1;
  v7 = *(unsigned __int16 *)(a2 + 96);
  if ( *(_DWORD *)(a2 + 24) != 47 )
    v4 = 40;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(_QWORD *)(a2 + 104);
  v10 = *(_QWORD *)(v8 + v4);
  v67 = *(_QWORD *)(v8 + v4 + 8);
  v11 = *(_QWORD *)(a2 + 48);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v76.m128i_i16[0] = *(_WORD *)(a2 + 96);
  v76.m128i_i64[1] = v9;
  v70 = v12;
  v57 = v13;
  v75.m128i_i16[0] = v12;
  v75.m128i_i64[1] = v13;
  if ( !(_WORD)v7 )
  {
    v61 = v7;
    v64 = v9;
    v16 = sub_30070B0((__int64)&v76);
    v15 = v64;
    v7 = v61;
    if ( !v16 )
    {
      v3 = v76.m128i_i64[0];
      v14 = 1;
      v84 = (unsigned __int16 *)v86;
      v85 = 0x500000000LL;
      goto LABEL_9;
    }
    if ( !sub_3007100((__int64)&v76) )
      goto LABEL_66;
    goto LABEL_88;
  }
  if ( (unsigned __int16)(v7 - 17) > 0xD3u )
  {
    if ( (_WORD)v7 == 5 )
    {
      v14 = 1;
      v7 = 6;
      v15 = 0;
      v84 = (unsigned __int16 *)v86;
      v85 = 0x500000000LL;
      goto LABEL_9;
    }
    v14 = 1;
    goto LABEL_61;
  }
  LOWORD(v7) = v7 - 176;
  if ( (unsigned __int16)v7 <= 0x34u )
  {
LABEL_88:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v76.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v76.m128i_i16[0] - 176) > 0x34u )
      {
        v48 = v76.m128i_u16[0] - 1;
        v14 = word_4456340[v48];
        goto LABEL_58;
      }
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      goto LABEL_57;
    }
LABEL_66:
    v14 = sub_3007130((__int64)&v76, a2);
    goto LABEL_67;
  }
LABEL_57:
  v46 = word_4456340;
  v47 = v76.m128i_u16[0];
  v48 = v76.m128i_u16[0] - 1;
  v14 = word_4456340[v48];
  if ( v76.m128i_i16[0] )
  {
LABEL_58:
    v15 = 0;
    v7 = (unsigned __int16)word_4456580[v48];
    goto LABEL_59;
  }
LABEL_67:
  v7 = (unsigned int)sub_3009970((__int64)&v76, a2, v47, (__int64)v46, v7);
  v15 = v49;
LABEL_59:
  v76.m128i_i16[0] = v7;
  v76.m128i_i64[1] = v15;
  switch ( (_WORD)v7 )
  {
    case 0xB:
      if ( v70 != 127 )
      {
LABEL_61:
        v15 = v76.m128i_i64[1];
LABEL_62:
        v3 = v76.m128i_i64[0];
        goto LABEL_63;
      }
      goto LABEL_80;
    case 0xA:
      if ( v70 != 138 )
        goto LABEL_61;
      goto LABEL_80;
    case 6:
      if ( v70 != 47 )
        goto LABEL_62;
LABEL_80:
      v7 = v75.m128i_u16[0];
      v76 = _mm_load_si128(&v75);
      v14 /= word_4456340[v70 - 1];
      goto LABEL_81;
  }
  if ( (_WORD)v7 != 5 )
    goto LABEL_61;
  if ( v70 == 37 )
    goto LABEL_80;
LABEL_81:
  if ( (_WORD)v7 != 5 )
    goto LABEL_61;
  v15 = 0;
  v7 = 6;
LABEL_63:
  v84 = (unsigned __int16 *)v86;
  v85 = 0x500000000LL;
  if ( v14 )
  {
LABEL_9:
    v17 = 0;
    v18 = v3;
    v19 = 5;
    v20 = v15;
    v21 = 0;
    v58 = v10;
    v22 = v7;
    while ( 1 )
    {
      LOWORD(v18) = v22;
      if ( v17 + 1 > v19 )
      {
        sub_C8D5F0((__int64)&v84, v86, v17 + 1, 0x10u, v7, v15);
        v17 = (unsigned int)v85;
      }
      v23 = &v84[8 * v17];
      ++v21;
      *(_QWORD *)v23 = v18;
      *((_QWORD *)v23 + 1) = v20;
      v17 = (unsigned int)(v85 + 1);
      LODWORD(v85) = v85 + 1;
      if ( v21 == v14 )
        break;
      v19 = HIDWORD(v85);
    }
    v6 = a1;
    v5 = a2;
    v10 = v58;
    if ( v17 + 1 > (unsigned __int64)HIDWORD(v85) )
    {
      sub_C8D5F0((__int64)&v84, v86, v17 + 1, 0x10u, 1, v15);
      v17 = (unsigned int)v85;
    }
    goto LABEL_16;
  }
  v17 = 0;
LABEL_16:
  v24 = &v84[8 * v17];
  *(_QWORD *)v24 = 1;
  v25 = v84;
  *((_QWORD *)v24 + 1) = 0;
  v26 = *(_QWORD **)(v6 + 64);
  LODWORD(v85) = v85 + 1;
  v62 = sub_33E5830(v26, v25, (unsigned int)v85);
  v27 = *(__int64 **)(v5 + 40);
  v59 = v28;
  v29 = *v27;
  v30 = *((_DWORD *)v27 + 2);
  v77.m128i_i64[0] = 0;
  v77.m128i_i32[2] = 0;
  v65 = v29;
  v78.m128i_i64[0] = 0;
  v78.m128i_i32[2] = 0;
  sub_36DF750(v6, v10, v67, (__int64)&v77, (__int64)&v78, a3);
  v31 = _mm_load_si128(&v77);
  v32 = *(_DWORD *)(v5 + 24);
  v83 = v30;
  v33 = _mm_load_si128(&v78);
  v81[0] = v31;
  v82 = v65;
  v81[1] = v33;
  if ( v32 > 552 )
    goto LABEL_42;
  if ( v32 <= 547 )
  {
    if ( v32 == 47 )
    {
      v79 = 0x100000978LL;
      v34 = sub_36D6650(v76.m128i_u16[0], 2428, 2425, 2426, 0x10000097BLL, 2423, 0x100000978LL);
    }
    else
    {
      if ( v32 != 298 )
        goto LABEL_42;
      v79 = 0x100000964LL;
      v34 = sub_36D6650(v76.m128i_u16[0], 2408, 2405, 2406, 0x100000967LL, 2403, 0x100000964LL);
    }
LABEL_22:
    v35 = v34;
    v36 = HIDWORD(v34);
    if ( !BYTE4(v34) )
      goto LABEL_23;
    goto LABEL_32;
  }
  switch ( v32 )
  {
    case 549:
      v79 = 0x100000970LL;
      v34 = sub_36D6650(v76.m128i_u16[0], 2420, 2417, 2418, 0x100000973LL, 2415, 0x100000970LL);
      goto LABEL_22;
    case 550:
      if ( v76.m128i_i16[0] == 12 )
      {
        v35 = 2421;
        goto LABEL_32;
      }
      if ( v76.m128i_i16[0] <= 0xCu )
      {
        if ( v76.m128i_i16[0] == 7 )
          goto LABEL_31;
LABEL_42:
        LODWORD(v36) = 0;
        goto LABEL_23;
      }
      if ( v76.m128i_i16[0] == 127 )
      {
LABEL_31:
        v35 = 2422;
        goto LABEL_32;
      }
      if ( v76.m128i_i16[0] > 0x7Fu )
      {
        if ( v76.m128i_i16[0] != 138 )
          goto LABEL_42;
        goto LABEL_31;
      }
      v35 = 2422;
      if ( v76.m128i_i16[0] != 37 && v76.m128i_i16[0] != 47 )
        goto LABEL_42;
LABEL_32:
      v38 = *(_QWORD *)(v5 + 80);
      v79 = v38;
      if ( v38 )
      {
        v68 = v35;
        sub_B96E90((__int64)&v79, v38, 1);
        v35 = v68;
      }
      v39 = *(_QWORD **)(v6 + 64);
      v80 = *(_DWORD *)(v5 + 72);
      v43 = sub_33E66D0(v39, v35, (__int64)&v79, v62, v59, v35, (unsigned __int64 *)v81, 3);
      if ( *(_DWORD *)(v5 + 24) == 298 )
      {
        v44 = v76.m128i_i16[0];
        if ( v70 == v76.m128i_i16[0] && (v57 == v76.m128i_i64[1] || v70) )
          goto LABEL_38;
        v45 = v5;
LABEL_69:
        v63 = sub_36E0C40(v70, v44, v45);
        if ( v14 )
        {
          v69 = v43;
          v66 = v5;
          WORD1(v5) = v56;
          v60 = v14;
          v50 = 0;
          do
          {
            v51 = v50;
            LOWORD(v5) = 7;
            ++v50;
            v71 = *(_QWORD **)(v6 + 64);
            v73 = v51 | v73 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v52 = sub_3400BD0((__int64)v71, 0, (__int64)&v79, (unsigned int)v5, 0, 1u, v31, 0);
            *((_QWORD *)&v55 + 1) = v73;
            *(_QWORD *)&v55 = v69;
            v72 = sub_33F77A0(v71, v63, (__int64)&v79, v75.m128i_u32[0], v75.m128i_i64[1], v53, v55, v52);
            sub_34161C0(*(_QWORD *)(v6 + 64), v66, v51, v72, 0);
            sub_3421DB0(v72);
          }
          while ( v50 != v60 );
          v43 = v69;
          v5 = v66;
        }
        goto LABEL_38;
      }
      v44 = v76.m128i_i16[0];
      if ( v76.m128i_i16[0] == v70 )
      {
        if ( v70 || v57 == v76.m128i_i64[1] || !(unsigned __int8)sub_3007030((__int64)&v75) )
          goto LABEL_38;
      }
      else
      {
        v40 = v70;
        if ( v70 )
        {
          if ( (unsigned __int16)(v70 - 10) > 6u
            && (unsigned __int16)(v70 - 126) > 0x31u
            && (unsigned __int16)(v70 - 208) > 0x14u )
          {
            goto LABEL_38;
          }
        }
        else
        {
          v44 = v76.m128i_i16[0];
          if ( !(unsigned __int8)sub_3007030((__int64)&v75) )
            goto LABEL_38;
        }
        if ( v44 )
        {
          if ( (unsigned __int16)(v44 - 10) > 6u
            && (unsigned __int16)(v44 - 126) > 0x31u
            && (unsigned __int16)(v44 - 208) > 0x14u )
          {
            goto LABEL_38;
          }
          v45 = 0;
          goto LABEL_69;
        }
      }
      v54 = sub_3007030((__int64)&v76);
      v44 = 0;
      v45 = 0;
      if ( v54 )
        goto LABEL_69;
LABEL_38:
      sub_34158F0(*(_QWORD *)(v6 + 64), v5, v43, v40, v41, v42);
      sub_3421DB0(v43);
      sub_33ECEA0(*(const __m128i **)(v6 + 64), v5);
      if ( v79 )
        sub_B91220((__int64)&v79, v79);
      LODWORD(v36) = 1;
LABEL_23:
      if ( v84 != (unsigned __int16 *)v86 )
        _libc_free((unsigned __int64)v84);
      return (unsigned int)v36;
    case 551:
      v79 = 0x10000097ELL;
      v34 = sub_36D6650(v76.m128i_u16[0], 2434, 2431, 2432, 0x100000981LL, 2429, 0x10000097ELL);
      goto LABEL_22;
    case 552:
      BYTE4(v79) = 0;
      BYTE4(v74) = 0;
      v34 = sub_36D6650(v76.m128i_u16[0], 2440, 2438, 2439, v74, 2437, v79);
      goto LABEL_22;
    default:
      v79 = 0x10000096ALL;
      v34 = sub_36D6650(v76.m128i_u16[0], 2414, 2411, 2412, 0x10000096DLL, 2409, 0x10000096ALL);
      goto LABEL_22;
  }
}
