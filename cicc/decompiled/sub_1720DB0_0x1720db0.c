// Function: sub_1720DB0
// Address: 0x1720db0
//
__int64 __fastcall sub_1720DB0(
        __m128i *a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  unsigned __int8 *v11; // r12
  __m128 v12; // xmm0
  __m128i v13; // xmm1
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // r15
  double v17; // xmm4_8
  double v18; // xmm5_8
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rcx
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rsi
  __int64 v29; // rdx
  int v30; // edi
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdx
  unsigned __int8 v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rax
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // r10
  __int64 v41; // r9
  __int64 v42; // rbx
  __int64 v43; // r11
  __int64 v44; // rax
  int v45; // r12d
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r11
  __int64 v49; // rax
  __int64 v50; // r11
  __int64 *v51; // r12
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rsi
  __int64 v57; // rsi
  __int64 v58; // r12
  unsigned __int8 *v59; // rsi
  int v60; // eax
  int v61; // eax
  __int64 *v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rdi
  int v66; // esi
  __int64 v67; // rdx
  __int64 **v68; // rcx
  __int64 v69; // rax
  char v70; // al
  int v71; // ecx
  void *v72; // rax
  int v73; // eax
  int v74; // eax
  __int64 *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  unsigned __int64 v78; // rdx
  double v79; // xmm4_8
  double v80; // xmm5_8
  __int64 v81; // rdx
  __int64 v82; // rdx
  int v83; // [rsp+4h] [rbp-ACh]
  __int64 v84; // [rsp+8h] [rbp-A8h]
  __int64 v85; // [rsp+8h] [rbp-A8h]
  __int64 v86; // [rsp+10h] [rbp-A0h]
  __int64 v87; // [rsp+10h] [rbp-A0h]
  __int64 *v88; // [rsp+10h] [rbp-A0h]
  int v89; // [rsp+10h] [rbp-A0h]
  __int64 v90; // [rsp+18h] [rbp-98h]
  __int64 v91; // [rsp+18h] [rbp-98h]
  __int64 v92; // [rsp+18h] [rbp-98h]
  __int64 v93; // [rsp+18h] [rbp-98h]
  __int64 v94; // [rsp+18h] [rbp-98h]
  __int64 v95; // [rsp+18h] [rbp-98h]
  __int64 v96; // [rsp+18h] [rbp-98h]
  __int64 v97; // [rsp+28h] [rbp-88h] BYREF
  __int64 v98[2]; // [rsp+30h] [rbp-80h] BYREF
  __int16 v99; // [rsp+40h] [rbp-70h]
  __m128 v100; // [rsp+50h] [rbp-60h] BYREF
  __m128i v101; // [rsp+60h] [rbp-50h]
  __int64 v102; // [rsp+70h] [rbp-40h]

  v10 = a2;
  v11 = (unsigned __int8 *)a2;
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v102 = a2;
  v13 = _mm_loadu_si128(a1 + 168);
  v100 = v12;
  v101 = v13;
  v14 = sub_15F24E0(a2);
  v15 = sub_13D1D30(*(unsigned __int8 **)(a2 - 48), *(_QWORD *)(a2 - 24), v14, &v100);
  if ( !v15 )
  {
    v20 = (__int64)sub_1707490(
                     (__int64)a1,
                     (unsigned __int8 *)a2,
                     *(double *)v12.m128_u64,
                     *(double *)v13.m128i_i64,
                     a5);
    if ( v20 )
      return v20;
    v21 = *(_QWORD *)(a2 - 48);
    v90 = *(_QWORD *)(a2 - 24);
    if ( sub_15F24C0(a2) )
    {
      v23 = *(_BYTE *)(v21 + 16);
      if ( v23 == 14 )
      {
        if ( *(void **)(v21 + 32) == sub_16982C0() )
        {
          v44 = *(_QWORD *)(v21 + 40);
          v24 = *(_BYTE *)(v44 + 26) & 7;
          if ( (_BYTE)v24 != 3 )
            goto LABEL_24;
          v26 = v44 + 8;
        }
        else
        {
          if ( (*(_BYTE *)(v21 + 50) & 7) != 3 )
            goto LABEL_24;
          v26 = v21 + 32;
        }
        if ( (*(_BYTE *)(v26 + 18) & 8) == 0 )
        {
LABEL_15:
          v27 = sub_15A1390(*(_QWORD *)v90, a2, v24, v25);
          v101.m128i_i16[0] = 257;
          v28 = (__int64 *)v27;
          v29 = v90;
          v30 = 14;
LABEL_16:
          v11 = (unsigned __int8 *)sub_15FB440(v30, v28, v29, (__int64)&v100, 0);
          sub_15F2530(v11, v10, 1);
          return (__int64)v11;
        }
      }
      else if ( *(_BYTE *)(*(_QWORD *)v21 + 8LL) == 16 && v23 <= 0x10u )
      {
        v31 = sub_15A1020((_BYTE *)v21, a2, *(_QWORD *)v21, v22);
        if ( v31 && (v86 = v31, *(_BYTE *)(v31 + 16) == 14) )
        {
          if ( *(void **)(v31 + 32) == sub_16982C0() )
          {
            v82 = *(_QWORD *)(v86 + 40);
            if ( (*(_BYTE *)(v82 + 26) & 7) != 3 )
              goto LABEL_24;
            v24 = v82 + 8;
          }
          else
          {
            v24 = v86 + 32;
            if ( (*(_BYTE *)(v86 + 50) & 7) != 3 )
              goto LABEL_24;
          }
          if ( (*(_BYTE *)(v24 + 18) & 8) == 0 )
            goto LABEL_15;
        }
        else
        {
          v83 = *(_QWORD *)(*(_QWORD *)v21 + 32LL);
          if ( !v83 )
            goto LABEL_15;
          LODWORD(v25) = 0;
          while ( 1 )
          {
            a2 = (unsigned int)v25;
            v89 = v25;
            v69 = sub_15A0A60(v21, v25);
            v24 = v69;
            if ( !v69 )
              break;
            v70 = *(_BYTE *)(v69 + 16);
            v85 = v24;
            v71 = v89;
            if ( v70 != 9 )
            {
              if ( v70 != 14 )
                break;
              v72 = sub_16982C0();
              v71 = v89;
              if ( *(void **)(v85 + 32) == v72 )
              {
                v81 = *(_QWORD *)(v85 + 40);
                if ( (*(_BYTE *)(v81 + 26) & 7) != 3 )
                  break;
                v24 = v81 + 8;
              }
              else
              {
                if ( (*(_BYTE *)(v85 + 50) & 7) != 3 )
                  break;
                v24 = v85 + 32;
              }
              if ( (*(_BYTE *)(v24 + 18) & 8) != 0 )
                break;
            }
            v25 = (unsigned int)(v71 + 1);
            if ( v83 == (_DWORD)v25 )
              goto LABEL_15;
          }
        }
      }
    }
LABEL_24:
    if ( !sub_15F24C0((__int64)v11) )
    {
      a2 = a1[167].m128i_i64[1];
      if ( !sub_14AB3F0(v21, (__int64 *)a2, 0, v32) )
      {
        v34 = *(_BYTE *)(v90 + 16);
        goto LABEL_28;
      }
    }
    v33 = *(_QWORD *)(v90 + 8);
    v34 = *(_BYTE *)(v90 + 16);
    if ( !v33 || *(_QWORD *)(v33 + 8) )
    {
LABEL_28:
      if ( *(_BYTE *)(v21 + 16) <= 0x10u && v34 == 79 )
      {
        a2 = (__int64)v11;
        v20 = sub_1707470((__int64)a1, v11, v90, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
        if ( v20 )
          return v20;
        v34 = *(_BYTE *)(v90 + 16);
      }
      if ( v34 <= 0x10u && v34 != 5 )
      {
        v101.m128i_i16[0] = 257;
        v29 = sub_15A2BF0((__int64 *)v90, a2, v33, v32, *(double *)v12.m128_u64, *(double *)v13.m128i_i64, a5);
LABEL_35:
        v28 = (__int64 *)v21;
        v30 = 12;
        goto LABEL_16;
      }
      v100.m128_u64[1] = (unsigned __int64)&v97;
      if ( (unsigned __int8)sub_171FB50((__int64)&v100, v90, (__int64)&v97, v32) )
      {
        v29 = v97;
        v101.m128i_i16[0] = 257;
        goto LABEL_35;
      }
      v100.m128_u64[1] = (unsigned __int64)&v97;
      v35 = *(_QWORD *)(v90 + 8);
      if ( v35 && !*(_QWORD *)(v35 + 8) )
      {
        v60 = *(unsigned __int8 *)(v90 + 16);
        if ( (unsigned __int8)v60 > 0x17u )
        {
          v61 = v60 - 24;
        }
        else
        {
          if ( (_BYTE)v60 != 5 )
            goto LABEL_39;
          v61 = *(unsigned __int16 *)(v90 + 18);
        }
        if ( v61 == 43 )
        {
          v62 = (__int64 *)sub_13CF970(v90);
          if ( (unsigned __int8)sub_171FB50((__int64)&v100, *v62, v63, v64) )
          {
            v65 = a1->m128i_i64[1];
            v66 = 43;
            v101.m128i_i16[0] = 257;
            v67 = v97;
            v68 = *(__int64 ***)v11;
LABEL_77:
            v29 = (__int64)sub_1708970(v65, v66, v67, v68, (__int64 *)&v100);
            v101.m128i_i16[0] = 257;
            goto LABEL_35;
          }
        }
      }
LABEL_39:
      v100.m128_u64[1] = (unsigned __int64)&v97;
      v36 = *(_QWORD *)(v90 + 8);
      if ( !v36 || *(_QWORD *)(v36 + 8) )
        goto LABEL_41;
      v73 = *(unsigned __int8 *)(v90 + 16);
      if ( (unsigned __int8)v73 > 0x17u )
      {
        v74 = v73 - 24;
      }
      else
      {
        if ( (_BYTE)v73 != 5 )
          goto LABEL_41;
        v74 = *(unsigned __int16 *)(v90 + 18);
      }
      if ( v74 == 44 )
      {
        v75 = (__int64 *)sub_13CF970(v90);
        if ( (unsigned __int8)sub_171FB50((__int64)&v100, *v75, v76, v77) )
        {
          v65 = a1->m128i_i64[1];
          v68 = *(__int64 ***)v11;
          v101.m128i_i16[0] = 257;
          v66 = 44;
          v67 = v97;
          goto LABEL_77;
        }
      }
LABEL_41:
      v37 = sub_1707FD0(a1, v11, v21, v90);
      if ( v37 )
        return sub_170E100(
                 a1->m128i_i64,
                 (__int64)v11,
                 (__int64)v37,
                 v12,
                 *(double *)v13.m128i_i64,
                 a5,
                 a6,
                 v38,
                 v39,
                 a9,
                 a10);
      if ( sub_15F24A0((__int64)v11) && sub_15F24C0((__int64)v11) )
      {
        v100 = (__m128)a1->m128i_u64[1];
        v78 = sub_171BFC0((__int64 *)&v100, (__int64)v11, (__m128i)v12, *(double *)v13.m128i_i64, a5);
        if ( v78 )
          return sub_170E100(a1->m128i_i64, (__int64)v11, v78, v12, *(double *)v13.m128i_i64, a5, a6, v79, v80, a9, a10);
      }
      return 0;
    }
    if ( v34 == 38 )
    {
      v32 = v90;
      v40 = *(_QWORD *)(v90 - 48);
      if ( !v40 )
        goto LABEL_28;
      v41 = *(_QWORD *)(v90 - 24);
      if ( !v41 )
        goto LABEL_28;
    }
    else
    {
      if ( v34 != 5 )
        goto LABEL_28;
      v32 = v90;
      if ( *(_WORD *)(v90 + 18) != 14 )
        goto LABEL_28;
      v33 = *(_DWORD *)(v90 + 20) & 0xFFFFFFF;
      a2 = 4 * v33;
      v32 = -3 * v33;
      v40 = *(_QWORD *)(v90 - 24 * v33);
      if ( !v40 )
        goto LABEL_28;
      v32 = 1 - v33;
      v33 = 3 * (1 - v33);
      v41 = *(_QWORD *)(v90 + 8 * v33);
      if ( !v41 )
        goto LABEL_28;
    }
    v97 = v41;
    v42 = a1->m128i_i64[1];
    v99 = 257;
    if ( *(_BYTE *)(v41 + 16) > 0x10u
      || *(_BYTE *)(v40 + 16) > 0x10u
      || (v84 = v40,
          v87 = v41,
          v91 = sub_15A2A30(
                  (__int64 *)0xE,
                  (__int64 *)v41,
                  v40,
                  0,
                  0,
                  *(double *)v12.m128_u64,
                  *(double *)v13.m128i_i64,
                  a5),
          (v43 = sub_14DBA30(v91, *(_QWORD *)(v42 + 96), 0)) == 0)
      && (v41 = v87, v40 = v84, (v43 = v91) == 0) )
    {
      v88 = (__int64 *)v41;
      v92 = v40;
      v45 = sub_15F24E0((__int64)v11);
      v101.m128i_i16[0] = 257;
      v46 = sub_15FB440(14, v88, v92, (__int64)&v100, 0);
      v47 = *(_QWORD *)(v42 + 32);
      v48 = v46;
      if ( v47 )
      {
        v93 = v46;
        sub_1625C10(v46, 3, v47);
        v48 = v93;
      }
      v94 = v48;
      sub_15F2440(v48, v45);
      v49 = *(_QWORD *)(v42 + 8);
      v50 = v94;
      if ( v49 )
      {
        v51 = *(__int64 **)(v42 + 16);
        sub_157E9D0(v49 + 40, v94);
        v50 = v94;
        v52 = *v51;
        v53 = *(_QWORD *)(v94 + 24);
        *(_QWORD *)(v94 + 32) = v51;
        v52 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v94 + 24) = v52 | v53 & 7;
        *(_QWORD *)(v52 + 8) = v94 + 24;
        *v51 = *v51 & 7 | (v94 + 24);
      }
      v54 = v50;
      v95 = v50;
      sub_164B780(v50, v98);
      v100.m128_u64[0] = v95;
      if ( !*(_QWORD *)(v42 + 80) )
        sub_4263D6(v54, v98, v55);
      (*(void (__fastcall **)(__int64, __m128 *))(v42 + 88))(v42 + 64, &v100);
      v56 = *(_QWORD *)v42;
      v43 = v95;
      if ( *(_QWORD *)v42 )
      {
        v100.m128_u64[0] = *(_QWORD *)v42;
        sub_1623A60((__int64)&v100, v56, 2);
        v43 = v95;
        v57 = *(_QWORD *)(v95 + 48);
        v58 = v95 + 48;
        if ( v57 )
        {
          sub_161E7C0(v95 + 48, v57);
          v43 = v95;
        }
        v59 = (unsigned __int8 *)v100.m128_u64[0];
        *(_QWORD *)(v43 + 48) = v100.m128_u64[0];
        if ( v59 )
        {
          v96 = v43;
          sub_1623210((__int64)&v100, v59, v58);
          v43 = v96;
        }
      }
    }
    v29 = v43;
    v101.m128i_i16[0] = 257;
    goto LABEL_35;
  }
  if ( !*(_QWORD *)(a2 + 8) )
    return 0;
  v16 = v15;
  sub_17205C0(a1->m128i_i64[0], a2);
  if ( a2 == v16 )
    v16 = sub_1599EF0(*(__int64 ***)a2);
  sub_164D160(a2, v16, v12, *(double *)v13.m128i_i64, a5, a6, v17, v18, a9, a10);
  return (__int64)v11;
}
