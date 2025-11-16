// Function: sub_340EC60
// Address: 0x340ec60
//
__int64 __fastcall sub_340EC60(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int128 a10)
{
  __int64 v10; // r10
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // r11
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  __m128i *v21; // rax
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int32 v26; // r8d
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // r15
  int v29; // ebx
  unsigned __int8 *v30; // rsi
  __int64 result; // rax
  __m128i *v32; // rax
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // r14
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rsi
  int v41; // eax
  int v42; // eax
  unsigned __int8 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rsi
  unsigned __int8 *v46; // r8
  __int64 v47; // r10
  __int64 v48; // r11
  unsigned __int8 **v49; // rax
  char v50; // al
  __int64 v51; // rdx
  __m128i v52; // xmm4
  __m128i v53; // xmm5
  __int64 v54; // rax
  __int64 v55; // rdi
  int v56; // eax
  __m128i v57; // xmm4
  __m128i v58; // xmm5
  int v59; // esi
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  __int64 v62; // rax
  __int16 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  _QWORD *v66; // rsi
  unsigned int v67; // eax
  __int64 v68; // rcx
  unsigned __int64 v69; // rax
  int v70; // esi
  __int64 v71; // rax
  __int64 *v72; // rcx
  __int64 v73; // rax
  bool v74; // al
  __int64 v75; // rax
  __int64 *v76; // rsi
  unsigned __int8 *v77; // r8
  __int64 v78; // r10
  __int64 v79; // r11
  __int64 v80; // rsi
  __int64 v81; // [rsp+8h] [rbp-168h]
  __int64 v82; // [rsp+10h] [rbp-160h]
  __int64 v84; // [rsp+18h] [rbp-158h]
  __int64 v85; // [rsp+18h] [rbp-158h]
  __int64 v86; // [rsp+18h] [rbp-158h]
  __int64 v87; // [rsp+20h] [rbp-150h]
  __int64 v88; // [rsp+20h] [rbp-150h]
  __int64 v89; // [rsp+20h] [rbp-150h]
  unsigned __int8 *v90; // [rsp+20h] [rbp-150h]
  __int64 v91; // [rsp+20h] [rbp-150h]
  __int64 v92; // [rsp+20h] [rbp-150h]
  __int64 v93; // [rsp+28h] [rbp-148h]
  unsigned __int8 *v94; // [rsp+28h] [rbp-148h]
  __int64 v95; // [rsp+28h] [rbp-148h]
  __int64 v98; // [rsp+28h] [rbp-148h]
  unsigned __int8 *v99; // [rsp+28h] [rbp-148h]
  __int64 v100; // [rsp+28h] [rbp-148h]
  __int64 v101; // [rsp+30h] [rbp-140h]
  int v103; // [rsp+30h] [rbp-140h]
  int v106; // [rsp+3Ch] [rbp-134h]
  __m128i v107; // [rsp+40h] [rbp-130h] BYREF
  __m128i v108; // [rsp+50h] [rbp-120h] BYREF
  __int64 *v109; // [rsp+68h] [rbp-108h] BYREF
  unsigned __int64 v110; // [rsp+70h] [rbp-100h] BYREF
  __int64 v111; // [rsp+78h] [rbp-F8h]
  __int64 v112; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v113; // [rsp+88h] [rbp-E8h]
  __m128i v114; // [rsp+90h] [rbp-E0h]
  __m128i v115; // [rsp+A0h] [rbp-D0h]
  unsigned __int8 *v116; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v117; // [rsp+B8h] [rbp-B8h]
  __m128i v118; // [rsp+C0h] [rbp-B0h] BYREF
  __m128i v119; // [rsp+D0h] [rbp-A0h]

  v10 = a5;
  v15 = _mm_loadu_si128((const __m128i *)&a9);
  v16 = _mm_loadu_si128((const __m128i *)&a10);
  v106 = a6;
  v17 = a7;
  v108 = v15;
  v18 = v15.m128i_i64[0];
  v107 = v16;
  v19 = v16.m128i_i64[0];
  if ( a2 > 0xEA )
  {
    if ( a2 - 458 <= 2 )
    {
LABEL_5:
      v20 = *(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8;
      if ( *(_WORD *)v20 == (_WORD)a4 && (*(_QWORD *)(v20 + 8) == a5 || (_WORD)a4) )
        return v17;
    }
LABEL_6:
    v101 = v17;
    v21 = sub_33ED250((__int64)a1, a4, v10);
    v22 = _mm_load_si128(&v108);
    v113 = a8;
    v23 = _mm_load_si128(&v107);
    v110 = (unsigned __int64)v21;
    v111 = v24;
    v112 = v101;
    v114 = v22;
    v115 = v23;
    if ( (_WORD)a4 != 262 )
    {
LABEL_17:
      v108.m128i_i64[0] = (__int64)&v112;
      v117 = 0x2000000000LL;
      v116 = (unsigned __int8 *)&v118;
      sub_33C9670((__int64)&v116, a2, v110, (unsigned __int64 *)&v112, 3, (__int64)&v112);
      v109 = 0;
      v36 = sub_33CCCF0((__int64)a1, (__int64)&v116, a3, (__int64 *)&v109);
      v37 = (__int64)v36;
      if ( v36 )
      {
        sub_33D00A0((__int64)v36, v106);
        result = v37;
        if ( v116 != (unsigned __int8 *)&v118 )
        {
          v107.m128i_i64[0] = 0;
          v108.m128i_i64[0] = v37;
          _libc_free((unsigned __int64)v116);
          return v108.m128i_i64[0];
        }
        return result;
      }
      v27 = sub_33E6540(a1, a2, *(_DWORD *)(a3 + 8), (__int64 *)a3, (__int64 *)&v110);
      v38 = v108.m128i_i64[0];
      *(_DWORD *)(v27 + 28) = v106;
      sub_33E4EC0((__int64)a1, v27, v38, 3);
      sub_C657C0(a1 + 65, (__int64 *)v27, v109, (__int64)off_4A367D0);
      if ( v116 != (unsigned __int8 *)&v118 )
        _libc_free((unsigned __int64)v116);
LABEL_15:
      sub_33CC420((__int64)a1, v27);
      return v27;
    }
    v25 = *(_QWORD *)a3;
    v26 = *(_DWORD *)(a3 + 8);
    v116 = (unsigned __int8 *)v25;
    if ( v25 )
    {
      v108.m128i_i32[0] = v26;
      sub_B96E90((__int64)&v116, v25, 1);
      v26 = v108.m128i_i32[0];
    }
    v27 = a1[52];
    v28 = v110;
    v29 = v111;
    if ( v27 )
    {
      a1[52] = *(_QWORD *)v27;
    }
    else
    {
      v68 = a1[53];
      a1[63] += 120LL;
      v69 = (v68 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( a1[54] >= v69 + 120 && v68 )
      {
        a1[53] = v69 + 120;
        if ( !v69 )
        {
          if ( v116 )
            sub_B91220((__int64)&v116, (__int64)v116);
          goto LABEL_14;
        }
        v27 = (v68 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      }
      else
      {
        v108.m128i_i32[0] = v26;
        v75 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
        v26 = v108.m128i_i32[0];
        v27 = v75;
      }
    }
    *(_QWORD *)v27 = 0;
    *(_QWORD *)(v27 + 8) = 0;
    *(_QWORD *)(v27 + 16) = 0;
    *(_DWORD *)(v27 + 24) = a2;
    *(_DWORD *)(v27 + 28) = 0;
    *(_WORD *)(v27 + 34) = -1;
    *(_DWORD *)(v27 + 36) = -1;
    *(_QWORD *)(v27 + 40) = 0;
    *(_QWORD *)(v27 + 48) = v28;
    *(_QWORD *)(v27 + 56) = 0;
    *(_DWORD *)(v27 + 64) = 0;
    *(_DWORD *)(v27 + 68) = v29;
    *(_DWORD *)(v27 + 72) = v26;
    v30 = v116;
    *(_QWORD *)(v27 + 80) = v116;
    if ( v30 )
      sub_B976B0((__int64)&v116, v30, v27 + 80);
    *(_QWORD *)(v27 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v27 + 32) = 0;
LABEL_14:
    sub_33E4EC0((__int64)a1, v27, (__int64)&v112, 3);
    goto LABEL_15;
  }
  if ( a2 <= 0x95 )
    goto LABEL_6;
  switch ( a2 )
  {
    case 0x96u:
    case 0x97u:
      v39 = *(_DWORD *)(a7 + 24);
      v40 = a7;
      if ( v39 != 12 )
      {
        v40 = 0;
        if ( v39 == 36 )
          v40 = a7;
      }
      v41 = *(_DWORD *)(v15.m128i_i64[0] + 24);
      if ( v41 != 12 && v41 != 36 )
        goto LABEL_6;
      v42 = *(_DWORD *)(v16.m128i_i64[0] + 24);
      if ( v42 != 12 && v42 != 36 )
        goto LABEL_6;
      if ( !v40 )
        goto LABEL_6;
      sub_9693D0((__int64)&v116, (_QWORD *)(*(_QWORD *)(v40 + 96) + 24LL));
      v87 = *(_QWORD *)(v15.m128i_i64[0] + 96);
      v93 = *(_QWORD *)(v16.m128i_i64[0] + 96);
      v43 = (unsigned __int8 *)sub_C33340();
      if ( a2 == 151 )
      {
        v76 = (__int64 *)(v87 + 24);
        v90 = v43;
        if ( v43 == v116 )
        {
          sub_C3F5C0((__int64)&v116, v76, 1u);
          v79 = a7;
          v78 = a5;
          v77 = v90;
        }
        else
        {
          sub_C3B950((__int64)&v116, (__int64)v76, 1);
          v77 = v90;
          v78 = a5;
          v79 = a7;
        }
        v86 = v79;
        v91 = v78;
        v80 = v93 + 24;
        v99 = v77;
        if ( v77 == v116 )
          sub_C3D800((__int64 *)&v116, v80, 1u);
        else
          sub_C3ADF0((__int64)&v116, v80, 1);
        v46 = v99;
        v47 = v91;
        v48 = v86;
      }
      else
      {
        v44 = v93 + 24;
        v45 = v87 + 24;
        v94 = v43;
        if ( v43 == v116 )
        {
          sub_C3F220(&v116, v45, v44, 1u);
          v48 = a7;
          v47 = a5;
          v46 = v94;
        }
        else
        {
          sub_C3B3E0((__int64)&v116, v45, v44, 1);
          v46 = v94;
          v47 = a5;
          v48 = a7;
        }
      }
      v49 = &v116;
      v95 = v48;
      if ( v46 == v116 )
        v49 = (unsigned __int8 **)v117;
      if ( (*((_BYTE *)v49 + 20) & 7) != 1 || (v88 = v47, v50 = sub_C33750((__int64)v116), v47 = v88, !v50) )
      {
        v107.m128i_i64[0] = sub_33FE6E0((__int64)a1, (__int64 *)&v116, a3, a4, v47, 0, v15);
        v108.m128i_i64[0] = v51;
        sub_91D830(&v116);
        return v107.m128i_i64[0];
      }
      v92 = v95;
      v100 = v47;
      sub_91D830(&v116);
      v10 = v100;
      v17 = v92;
      goto LABEL_6;
    case 0x9Cu:
      v52 = _mm_load_si128(&v108);
      v53 = _mm_load_si128(&v107);
      v116 = (unsigned __int8 *)a7;
      v117 = a8;
      v118 = v52;
      v119 = v53;
      result = (__int64)sub_33F2070(a4, a5, (char *)&v116, 3, a1);
      v10 = a5;
      v17 = a7;
      if ( !result )
        goto LABEL_6;
      return result;
    case 0x9Du:
      v59 = *(_DWORD *)(v16.m128i_i64[0] + 24);
      if ( v59 != 35 && v59 != 11 )
        goto LABEL_62;
      v62 = *(_QWORD *)(a7 + 48) + 16LL * (unsigned int)a8;
      v63 = *(_WORD *)v62;
      v64 = *(_QWORD *)(v62 + 8);
      LOWORD(v112) = v63;
      v113 = v64;
      if ( v63 )
      {
        if ( (unsigned __int16)(v63 - 17) > 0x9Eu )
          goto LABEL_62;
      }
      else
      {
        v81 = v10;
        v85 = v64;
        v74 = sub_30070D0((__int64)&v112);
        v19 = v16.m128i_i64[0];
        v63 = 0;
        v64 = v85;
        v18 = v15.m128i_i64[0];
        v10 = v81;
        v17 = a7;
        if ( !v74 )
          goto LABEL_62;
      }
      v65 = *(_QWORD *)(v19 + 96);
      v66 = *(_QWORD **)(v65 + 24);
      if ( *(_DWORD *)(v65 + 32) > 0x40u )
        v66 = (_QWORD *)*v66;
      v98 = v10;
      v82 = v17;
      v84 = v18;
      v89 = v19;
      LOWORD(v116) = v63;
      v117 = v64;
      v67 = sub_3281500(&v116, (__int64)v66);
      v10 = v98;
      if ( v67 <= (unsigned __int64)v66 )
        return sub_3288990((__int64)a1, a4, v10);
      v17 = v82;
      v18 = v84;
      v59 = *(_DWORD *)(v89 + 24);
LABEL_62:
      if ( v59 == 51 )
        return sub_3288990((__int64)a1, a4, v10);
      if ( *(_DWORD *)(v18 + 24) == 51 )
        return v17;
      goto LABEL_6;
    case 0x9Fu:
      v60 = _mm_load_si128(&v108);
      v61 = _mm_load_si128(&v107);
      v116 = (unsigned __int8 *)a7;
      v117 = a8;
      v118 = v60;
      v119 = v61;
      result = (__int64)sub_33FC250(a3, a4, a5, (char *)&v116, 3, a1, v15);
      v10 = a5;
      v17 = a7;
      if ( !result )
        goto LABEL_6;
      return result;
    case 0xA0u:
      if ( *(_DWORD *)(a7 + 24) != 51 )
      {
        v54 = *(_QWORD *)(v15.m128i_i64[0] + 48) + 16LL * v108.m128i_u32[2];
        if ( (_WORD)a4 != *(_WORD *)v54 )
          goto LABEL_6;
        if ( *(_QWORD *)(v54 + 8) != a5 && !(_WORD)a4 )
        {
          v32 = sub_33ED250((__int64)a1, a4, a5);
          v33 = _mm_load_si128(&v108);
          v113 = a8;
          v34 = _mm_load_si128(&v107);
          v110 = (unsigned __int64)v32;
          v111 = v35;
          v112 = a7;
          v114 = v33;
          v115 = v34;
          goto LABEL_17;
        }
        return v108.m128i_i64[0];
      }
      v70 = *(_DWORD *)(v15.m128i_i64[0] + 24);
      if ( v70 == 51 )
        return sub_3288990((__int64)a1, a4, v10);
      v71 = *(_QWORD *)(v15.m128i_i64[0] + 48) + 16LL * v108.m128i_u32[2];
      if ( (_WORD)a4 == *(_WORD *)v71 && ((_WORD)a4 || *(_QWORD *)(v71 + 8) == a5) )
        return v108.m128i_i64[0];
      if ( v70 != 161 )
        goto LABEL_6;
      v72 = *(__int64 **)(v15.m128i_i64[0] + 40);
      if ( v16.m128i_i64[0] != v72[5] )
        goto LABEL_6;
      if ( *((_DWORD *)v72 + 12) != v107.m128i_i32[2] )
        goto LABEL_6;
      v73 = *(_QWORD *)(*v72 + 48) + 16LL * *((unsigned int *)v72 + 2);
      if ( *(_WORD *)v73 != (_WORD)a4 || *(_QWORD *)(v73 + 8) != a5 && !(_WORD)a4 )
        goto LABEL_6;
      result = *v72;
      break;
    case 0xA5u:
      BUG();
    case 0xA6u:
      v55 = *(_QWORD *)(v16.m128i_i64[0] + 96);
      if ( *(_DWORD *)(v55 + 32) > 0x40u )
      {
        v103 = *(_DWORD *)(v55 + 32);
        v56 = sub_C444A0(v55 + 24);
        v10 = a5;
        v17 = a7;
        if ( v103 != v56 )
          goto LABEL_6;
        return v17;
      }
      if ( !*(_QWORD *)(v55 + 24) )
        return v17;
      goto LABEL_6;
    case 0xABu:
      if ( *(_DWORD *)(a7 + 24) != 51 && *(_DWORD *)(v15.m128i_i64[0] + 24) != 51 )
        goto LABEL_6;
      return v107.m128i_i64[0];
    case 0xCDu:
    case 0xCEu:
      result = sub_33E28A0(
                 (__int64)a1,
                 a7,
                 a8,
                 v108.m128i_i64[0],
                 v108.m128i_u64[1],
                 a6,
                 v107.m128i_i64[0],
                 v107.m128i_i64[1]);
      v17 = a7;
      v10 = a5;
      if ( !result )
        goto LABEL_6;
      return result;
    case 0xD0u:
      result = sub_340F940(
                 (_DWORD)a1,
                 a4,
                 a5,
                 a7,
                 a8,
                 *(_DWORD *)(v16.m128i_i64[0] + 96),
                 v108.m128i_i64[0],
                 v108.m128i_i64[1],
                 a3);
      if ( !result )
      {
        v116 = (unsigned __int8 *)a7;
        v57 = _mm_load_si128(&v108);
        v58 = _mm_load_si128(&v107);
        v117 = a8;
        v118 = v57;
        v119 = v58;
        result = (__int64)sub_3402EA0(
                            (__int64)a1,
                            208,
                            (unsigned __int64 *)a3,
                            a4,
                            a5,
                            0,
                            v15,
                            (unsigned int *)&v116,
                            3);
        v10 = a5;
        v17 = a7;
        if ( !result )
          goto LABEL_6;
      }
      return result;
    case 0xEAu:
      goto LABEL_5;
    default:
      goto LABEL_6;
  }
  return result;
}
