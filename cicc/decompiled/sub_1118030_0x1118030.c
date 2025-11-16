// Function: sub_1118030
// Address: 0x1118030
//
__int64 __fastcall sub_1118030(const __m128i *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  unsigned __int16 v4; // r12
  bool v6; // r15
  char *v7; // rsi
  _BYTE *v8; // rax
  __int16 v9; // r15
  char v10; // dl
  unsigned int v11; // r15d
  bool v12; // al
  __m128i v13; // xmm1
  unsigned __int64 v14; // xmm2_8
  __int64 v15; // rax
  __m128i v16; // xmm3
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r13
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  unsigned __int64 v23; // xmm6_8
  __m128i v24; // xmm7
  int v25; // eax
  __int64 v26; // r11
  __int64 v27; // r13
  _QWORD *v28; // rax
  __int64 v29; // r11
  __int64 v30; // rdx
  _BYTE *v31; // rax
  __int64 v32; // r13
  __m128i v33; // xmm4
  __m128i v34; // xmm5
  unsigned __int64 v35; // xmm6_8
  __m128i v36; // xmm7
  __m128i v37; // xmm5
  unsigned __int64 v38; // xmm6_8
  __int64 v39; // rax
  __m128i v40; // xmm7
  int v41; // eax
  unsigned int v42; // eax
  __int64 v43; // r14
  _QWORD *v44; // rax
  unsigned int v45; // edx
  _BYTE *v46; // rax
  unsigned int v47; // edx
  bool v48; // al
  __m128i v49; // xmm5
  unsigned __int64 v50; // xmm6_8
  __int64 v51; // rax
  __m128i v52; // xmm7
  __int64 v53; // r12
  unsigned int v56; // edx
  __m128i v57; // xmm4
  __m128i v58; // xmm5
  unsigned __int64 v59; // xmm6_8
  __m128i v60; // xmm7
  int v61; // eax
  __int64 v62; // r11
  __int64 v63; // r14
  _QWORD *v64; // rax
  __m128i *v65; // r8
  __int64 v66; // rcx
  __int64 v67; // rdx
  __m128i v68; // xmm5
  unsigned __int64 v69; // xmm6_8
  __int64 v70; // rax
  __m128i v71; // xmm7
  bool v72; // al
  __int64 v73; // r11
  __int64 v74; // r12
  unsigned int v77; // edx
  char v78; // al
  __int64 v79; // r12
  _QWORD *v80; // [rsp+8h] [rbp-118h]
  _QWORD *v81; // [rsp+8h] [rbp-118h]
  __int64 v82; // [rsp+8h] [rbp-118h]
  __int64 v83; // [rsp+8h] [rbp-118h]
  __int64 v84; // [rsp+10h] [rbp-110h]
  __int64 v85; // [rsp+18h] [rbp-108h]
  _QWORD *v86; // [rsp+18h] [rbp-108h]
  _QWORD *v87; // [rsp+18h] [rbp-108h]
  int v88; // [rsp+20h] [rbp-100h]
  __int64 v89; // [rsp+28h] [rbp-F8h]
  __int64 v90; // [rsp+28h] [rbp-F8h]
  __int64 v91; // [rsp+28h] [rbp-F8h]
  __int64 v92; // [rsp+28h] [rbp-F8h]
  unsigned int v93; // [rsp+28h] [rbp-F8h]
  _QWORD *v94; // [rsp+28h] [rbp-F8h]
  _QWORD *v95; // [rsp+28h] [rbp-F8h]
  unsigned int v96; // [rsp+28h] [rbp-F8h]
  __int64 v97[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v98; // [rsp+40h] [rbp-E0h] BYREF
  unsigned int v99; // [rsp+48h] [rbp-D8h]
  __int64 v100; // [rsp+50h] [rbp-D0h] BYREF
  unsigned int v101; // [rsp+58h] [rbp-C8h]
  __int64 v102; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v103; // [rsp+68h] [rbp-B8h]
  __int64 v104[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v105; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v106; // [rsp+88h] [rbp-98h]
  __int16 v107; // [rsp+90h] [rbp-90h]
  __m128i v108; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v109; // [rsp+B0h] [rbp-70h]
  unsigned __int64 v110; // [rsp+C0h] [rbp-60h]
  __int64 v111; // [rsp+C8h] [rbp-58h]
  __m128i v112; // [rsp+D0h] [rbp-50h]
  __int64 v113; // [rsp+E0h] [rbp-40h]

  result = 0;
  v3 = *(_QWORD *)(a2 - 32);
  v4 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( *(_BYTE *)v3 > 0x15u )
    return result;
  v6 = sub_AC30F0(*(_QWORD *)(a2 - 32));
  if ( !v6 )
  {
    if ( *(_BYTE *)v3 == 17 )
    {
      v11 = *(_DWORD *)(v3 + 32);
      if ( v11 <= 0x40 )
        v12 = *(_QWORD *)(v3 + 24) == 0;
      else
        v12 = v11 == (unsigned int)sub_C444A0(v3 + 24);
      if ( !v12 )
        return 0;
      if ( v4 != 38 )
        goto LABEL_5;
      goto LABEL_4;
    }
    v30 = *(_QWORD *)(v3 + 8);
    v91 = v30;
    if ( (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17 > 1 )
      return 0;
    v31 = sub_AD7630(v3, 0, v30);
    if ( !v31 || *v31 != 17 )
    {
      if ( *(_BYTE *)(v91 + 8) == 17 )
      {
        v88 = *(_DWORD *)(v91 + 32);
        if ( v88 )
        {
          v45 = 0;
          while ( 1 )
          {
            v96 = v45;
            v46 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v3, v45);
            if ( !v46 )
              break;
            v47 = v96;
            if ( *v46 != 13 )
            {
              if ( *v46 != 17 )
                break;
              v48 = sub_9867B0((__int64)(v46 + 24));
              v47 = v96;
              v6 = v48;
              if ( !v48 )
                break;
            }
            v45 = v47 + 1;
            if ( v88 == v45 )
              goto LABEL_34;
          }
        }
      }
      return 0;
    }
    v6 = sub_9867B0((__int64)(v31 + 24));
LABEL_34:
    if ( !v6 )
      return 0;
  }
  if ( v4 != 38 )
    goto LABEL_5;
LABEL_4:
  v7 = *(char **)(a2 - 64);
  v108.m128i_i64[0] = (__int64)&v100;
  v108.m128i_i64[1] = (__int64)v104;
  if ( !sub_10075D0(&v108, v7) )
    goto LABEL_5;
  v13 = _mm_loadu_si128(a1 + 7);
  v14 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v15 = a1[10].m128i_i64[0];
  v16 = _mm_loadu_si128(a1 + 9);
  v108 = _mm_loadu_si128(a1 + 6);
  v110 = v14;
  v113 = v15;
  v111 = a2;
  v109 = v13;
  v112 = v16;
  if ( (unsigned __int8)sub_9B6330(v100, &v108, 0) )
  {
    v17 = *(_QWORD *)(a2 - 32);
    LOWORD(v110) = 257;
    result = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( !result )
      return result;
    v18 = v104[0];
    v19 = v17;
    goto LABEL_20;
  }
  v49 = _mm_loadu_si128(a1 + 7);
  v50 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v51 = a1[10].m128i_i64[0];
  v52 = _mm_loadu_si128(a1 + 9);
  v108 = _mm_loadu_si128(a1 + 6);
  v110 = v50;
  v113 = v51;
  v111 = a2;
  v109 = v49;
  v112 = v52;
  if ( (unsigned __int8)sub_9B6330(v104[0], &v108, 0) )
  {
    v53 = *(_QWORD *)(a2 - 32);
    LOWORD(v110) = 257;
    result = (__int64)sub_BD2C40(72, unk_3F10FD0);
    if ( !result )
      return result;
    v18 = v100;
    v19 = v53;
LABEL_20:
    v89 = result;
    sub_1113300(result, 38, v18, v19, (__int64)&v108);
    return v89;
  }
LABEL_5:
  if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
  {
    result = sub_1112E90(a1, a2);
    if ( result )
      return result;
  }
  v8 = *(_BYTE **)(a2 - 64);
  v9 = v4;
  v10 = *v8;
  if ( *v8 != 51 )
  {
LABEL_9:
    if ( v10 == 46 )
    {
      v20 = *((_QWORD *)v8 - 8);
      if ( v20 )
      {
        if ( *((_QWORD *)v8 - 4) && (unsigned int)v4 - 32 <= 1 )
        {
          v21 = _mm_loadu_si128(a1 + 6);
          v22 = _mm_loadu_si128(a1 + 7);
          v90 = *((_QWORD *)v8 - 4);
          v23 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v113 = a1[10].m128i_i64[0];
          v24 = _mm_loadu_si128(a1 + 9);
          v110 = v23;
          v108 = v21;
          v111 = a2;
          v109 = v22;
          v112 = v24;
          sub_9AC330((__int64)v97, v20, 0, &v108);
          v25 = v99;
          v26 = v90;
          if ( v99 <= 0x40 )
          {
            _RDX = v98;
            __asm { tzcnt   rcx, rdx }
            v56 = 64;
            if ( v98 )
              v56 = _RCX;
            if ( v99 > v56 )
              v25 = v56;
          }
          else
          {
            v25 = sub_C44590((__int64)&v98);
            v26 = v90;
          }
          if ( !v25 )
          {
            v85 = v26;
            LOWORD(v110) = 257;
            v27 = *(_QWORD *)(a2 - 32);
            v28 = sub_BD2C40(72, unk_3F10FD0);
            if ( v28 )
            {
              v29 = v85;
              v86 = v28;
              sub_1113300((__int64)v28, v4, v29, v27, (__int64)&v108);
              v28 = v86;
            }
            goto LABEL_29;
          }
          v57 = _mm_loadu_si128(a1 + 6);
          v58 = _mm_loadu_si128(a1 + 7);
          v84 = v26;
          v59 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
          v113 = a1[10].m128i_i64[0];
          v60 = _mm_loadu_si128(a1 + 9);
          v110 = v59;
          v108 = v57;
          v111 = a2;
          v109 = v58;
          v112 = v60;
          sub_9AC330((__int64)&v100, v26, 0, &v108);
          v61 = v103;
          v62 = v84;
          if ( v103 <= 0x40 )
          {
            _RDX = v102;
            __asm { tzcnt   rcx, rdx }
            v77 = 64;
            if ( v102 )
              v77 = _RCX;
            if ( v103 > v77 )
              v61 = v77;
          }
          else
          {
            v61 = sub_C44590((__int64)&v102);
            v62 = v84;
          }
          if ( v61 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(a2 - 64) + 1LL) & 2) == 0
              && ((*(_BYTE *)(*(_QWORD *)(a2 - 64) + 1LL) >> 1) & 2) == 0 )
            {
              goto LABEL_92;
            }
            v68 = _mm_loadu_si128(a1 + 7);
            v82 = v62;
            v69 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
            v70 = a1[10].m128i_i64[0];
            v71 = _mm_loadu_si128(a1 + 9);
            v108 = _mm_loadu_si128(a1 + 6);
            v110 = v69;
            v113 = v70;
            v111 = a2;
            v109 = v68;
            v112 = v71;
            v72 = sub_9867B0((__int64)&v98);
            v73 = v82;
            if ( !v72 || (v78 = sub_9B6260(v20, &v108, 0), v73 = v82, v78) )
            {
              v83 = v73;
              v74 = *(_QWORD *)(a2 - 32);
              v107 = 257;
              v64 = sub_BD2C40(72, unk_3F10FD0);
              if ( !v64 )
                goto LABEL_73;
              v65 = (__m128i *)v104;
              v66 = v74;
              v67 = v83;
              goto LABEL_72;
            }
            if ( sub_9867B0((__int64)&v102) && !(unsigned __int8)sub_9B6260(v82, &v108, 0) )
            {
LABEL_92:
              sub_969240(&v102);
              sub_969240(&v100);
              sub_969240(&v98);
              sub_969240(v97);
              return 0;
            }
            v79 = *(_QWORD *)(a2 - 32);
            v107 = 257;
            v64 = sub_BD2C40(72, unk_3F10FD0);
            if ( !v64 )
              goto LABEL_73;
            v65 = (__m128i *)v104;
            v66 = v79;
          }
          else
          {
            v63 = *(_QWORD *)(a2 - 32);
            LOWORD(v110) = 257;
            v64 = sub_BD2C40(72, unk_3F10FD0);
            if ( !v64 )
            {
LABEL_73:
              v81 = v64;
              sub_969240(&v102);
              sub_969240(&v100);
              v28 = v81;
LABEL_29:
              v87 = v28;
              sub_969240(&v98);
              sub_969240(v97);
              return (__int64)v87;
            }
            v65 = &v108;
            v66 = v63;
          }
          v67 = v20;
LABEL_72:
          v80 = v64;
          sub_1113300((__int64)v64, v9, v67, v66, (__int64)v65);
          v64 = v80;
          goto LABEL_73;
        }
      }
    }
    return 0;
  }
  v32 = *((_QWORD *)v8 - 8);
  if ( !v32 || !*((_QWORD *)v8 - 4) || (unsigned int)v4 - 32 > 1 )
    return 0;
  v33 = _mm_loadu_si128(a1 + 6);
  v34 = _mm_loadu_si128(a1 + 7);
  v92 = *((_QWORD *)v8 - 4);
  v35 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v113 = a1[10].m128i_i64[0];
  v36 = _mm_loadu_si128(a1 + 9);
  v110 = v35;
  v108 = v33;
  v109 = v34;
  v112 = v36;
  v111 = a2;
  sub_9AC330((__int64)&v100, v32, 0, &v108);
  v37 = _mm_loadu_si128(a1 + 7);
  v38 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v39 = a1[10].m128i_i64[0];
  v108 = _mm_loadu_si128(a1 + 6);
  v40 = _mm_loadu_si128(a1 + 9);
  v110 = v38;
  v113 = v39;
  v111 = a2;
  v109 = v37;
  v112 = v40;
  sub_9AC330((__int64)v104, v92, 0, &v108);
  v93 = v101;
  if ( v101 > 0x40 )
    v41 = sub_C44630((__int64)&v100);
  else
    v41 = sub_39FAC40(v100);
  if ( v93 - v41 != 1 || (v106 > 0x40 ? (v42 = sub_C44630((__int64)&v105)) : (v42 = sub_39FAC40(v105)), v42 <= 1) )
  {
    sub_969240(&v105);
    sub_969240(v104);
    sub_969240(&v102);
    sub_969240(&v100);
    v8 = *(_BYTE **)(a2 - 64);
    v10 = *v8;
    goto LABEL_9;
  }
  LOWORD(v110) = 257;
  v43 = *(_QWORD *)(a2 - 32);
  v44 = sub_BD2C40(72, unk_3F10FD0);
  if ( v44 )
  {
    v94 = v44;
    sub_1113300((__int64)v44, v4, v32, v43, (__int64)&v108);
    v44 = v94;
  }
  v95 = v44;
  sub_969240(&v105);
  sub_969240(v104);
  sub_969240(&v102);
  sub_969240(&v100);
  return (__int64)v95;
}
