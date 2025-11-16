// Function: sub_D05650
// Address: 0xd05650
//
__int64 __fastcall sub_D05650(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  unsigned __int8 *v5; // rbx
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // r12d
  int v9; // eax
  unsigned __int8 *v10; // rdx
  unsigned __int8 *v11; // r12
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // edx
  unsigned __int8 *v18; // r12
  __int64 v19; // rax
  signed __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // r14
  __int64 v24; // rcx
  unsigned __int64 v25; // rdi
  _QWORD *v26; // r12
  unsigned __int64 v27; // rax
  __int64 v28; // r13
  unsigned __int64 v29; // rax
  char **v30; // r12
  int v31; // eax
  bool v32; // al
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // r12
  int v39; // ecx
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rbx
  __int64 i; // r13
  __m128i v44; // xmm1
  bool v45; // cc
  char v46; // al
  __int64 v47; // rdi
  int v48; // eax
  __int64 v49; // rcx
  __int64 v50; // rax
  int v51; // r12d
  __int64 v52; // rdi
  unsigned __int64 v53; // rax
  __int64 v54; // rdx
  unsigned int v55; // eax
  unsigned __int8 v56; // dl
  char v57; // r13
  unsigned int v58; // eax
  char v59; // r12
  char v60; // dl
  __int32 v61; // edi
  char v62; // r12
  unsigned int v63; // esi
  bool v64; // cf
  unsigned int v65; // eax
  __m128i v66; // xmm2
  unsigned int v67; // eax
  __int64 v68; // rax
  __int64 v69; // r13
  __int64 v70; // rbx
  unsigned __int8 *v71; // r12
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  __m128i v77; // xmm3
  unsigned __int64 v78; // rcx
  unsigned __int64 v79; // rsi
  __int64 v80; // r8
  __m128i *v81; // r12
  int v82; // eax
  __m128i *v83; // r13
  __m128i v84; // xmm4
  unsigned int v85; // eax
  __int64 v86; // rdi
  __int32 v87; // eax
  __m128i v88; // xmm5
  __int64 v89; // rdi
  __int8 *v90; // r12
  unsigned int v91; // [rsp+10h] [rbp-110h]
  __int64 v92; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v93; // [rsp+18h] [rbp-108h]
  unsigned __int8 *v94; // [rsp+20h] [rbp-100h]
  __int64 v95; // [rsp+28h] [rbp-F8h]
  int v96; // [rsp+30h] [rbp-F0h]
  unsigned int v97; // [rsp+34h] [rbp-ECh]
  unsigned __int8 **v99; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int64 v100; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v101; // [rsp+50h] [rbp-D0h]
  __int64 v102; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v103; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v104; // [rsp+68h] [rbp-B8h]
  __m128i v105; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v106; // [rsp+80h] [rbp-A0h]
  unsigned __int64 v107; // [rsp+88h] [rbp-98h] BYREF
  unsigned int v108; // [rsp+90h] [rbp-90h]
  __int64 v109; // [rsp+98h] [rbp-88h] BYREF
  unsigned int v110; // [rsp+A0h] [rbp-80h]
  __int16 v111; // [rsp+A8h] [rbp-78h]
  __m128i v112; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v113; // [rsp+C0h] [rbp-60h]
  unsigned __int64 v114; // [rsp+C8h] [rbp-58h] BYREF
  __int128 v115; // [rsp+D0h] [rbp-50h]
  __int64 v116; // [rsp+E0h] [rbp-40h]
  __int16 v117; // [rsp+E8h] [rbp-38h]

  v3 = 0;
  v5 = a2;
  if ( *a2 >= 0x1Du )
    v3 = a2;
  v94 = v3;
  v6 = sub_AE43F0(a3, *((_QWORD *)a2 + 1));
  *(_DWORD *)(a1 + 16) = 1;
  v7 = 0;
  v8 = v6;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  v95 = a1 + 8;
  *(_DWORD *)(a1 + 264) = 7;
  v112.m128i_i32[2] = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690((__int64)&v112, 0, 0);
    if ( *(_DWORD *)(a1 + 16) > 0x40u )
    {
      v13 = *(_QWORD *)(a1 + 8);
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    v7 = v112.m128i_i64[0];
    v6 = v112.m128i_u32[2];
  }
  *(_QWORD *)(a1 + 8) = v7;
  *(_DWORD *)(a1 + 16) = v6;
  v96 = 6;
  v97 = v8;
  while ( 1 )
  {
    v9 = *v5;
    if ( (unsigned __int8)v9 <= 0x1Cu )
      break;
    if ( (unsigned int)(unsigned __int8)v9 - 78 <= 1 )
      goto LABEL_11;
    if ( (_BYTE)v9 == 63 )
      goto LABEL_29;
    if ( (_BYTE)v9 == 84 )
    {
      if ( (*((_DWORD *)v5 + 1) & 0x7FFFFFF) != 1 )
        goto LABEL_14;
      v5 = (unsigned __int8 *)**((_QWORD **)v5 - 1);
    }
    else
    {
      v14 = (unsigned int)(v9 - 34);
      if ( (unsigned __int8)v14 > 0x33u )
        goto LABEL_14;
      v15 = 0x8000000000041LL;
      if ( !_bittest64(&v15, v14) )
        goto LABEL_14;
      v16 = sub_98AC40((__int64)v5, 0);
      if ( !v16 )
        goto LABEL_14;
      v5 = (unsigned __int8 *)v16;
    }
LABEL_25:
    if ( !--v96 )
      goto LABEL_14;
  }
  if ( (_BYTE)v9 != 5 )
  {
    if ( (_BYTE)v9 != 1 || (unsigned __int8)sub_B2F6B0((__int64)v5) )
      goto LABEL_14;
    v5 = (unsigned __int8 *)*((_QWORD *)v5 - 4);
    goto LABEL_25;
  }
  v17 = *((unsigned __int16 *)v5 + 1);
  if ( (unsigned int)(v17 - 49) <= 1 )
  {
LABEL_11:
    if ( (v5[7] & 0x40) != 0 )
      v10 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
    else
      v10 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
    v11 = *(unsigned __int8 **)v10;
    if ( (unsigned int)sub_AE43F0(a3, *(_QWORD *)(*(_QWORD *)v10 + 8LL)) != v97 )
      goto LABEL_14;
    v5 = v11;
    goto LABEL_25;
  }
  if ( (_WORD)v17 != 34 )
    goto LABEL_14;
LABEL_29:
  *(_DWORD *)(a1 + 264) &= v5[1] >> 1;
  if ( (v5[7] & 0x40) != 0 )
    v18 = (unsigned __int8 *)*((_QWORD *)v5 - 1);
  else
    v18 = &v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
  v19 = sub_BB5290((__int64)v5);
  v99 = (unsigned __int8 **)(v18 + 32);
  v20 = v19 & 0xFFFFFFFFFFFFFFF9LL | 4;
  v100 = v20;
  v21 = *((_DWORD *)v5 + 1) & 0x7FFFFFF;
  v22 = 32 * (1 - v21);
  if ( v5 == &v5[v22] )
    goto LABEL_145;
  v93 = v5;
  v23 = (__int64 *)&v5[v22];
  while ( 2 )
  {
    v24 = *v23;
    if ( v20 )
    {
      if ( (v20 & 6) == 0 )
      {
        v25 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v26 = *(_QWORD **)(v24 + 24);
          if ( *(_DWORD *)(v24 + 32) > 0x40u )
            v26 = (_QWORD *)*v26;
          v27 = v20 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (_DWORD)v26 )
          {
            v74 = 16LL * (unsigned int)v26 + sub_AE4AC0(a3, v20 & 0xFFFFFFFFFFFFFFF8LL) + 24;
            v75 = *(_QWORD *)v74;
            LOBYTE(v74) = *(_BYTE *)(v74 + 8);
            v112.m128i_i64[0] = v75;
            v112.m128i_i8[8] = v74;
            v76 = sub_CA1930(&v112);
            sub_C46A40(v95, v76);
            v20 = v100;
            v25 = v100 & 0xFFFFFFFFFFFFFFF8LL;
            v27 = v100 & 0xFFFFFFFFFFFFFFF8LL;
          }
          v23 += 4;
          if ( !v20 )
            goto LABEL_78;
LABEL_44:
          v28 = (v20 >> 1) & 3;
          if ( v28 == 2 )
          {
            if ( v25 )
              goto LABEL_46;
LABEL_78:
            v27 = sub_BCBAE0(v25, *v99, v22);
          }
          else
          {
            if ( v28 != 1 || !v25 )
              goto LABEL_78;
            v27 = *(_QWORD *)(v25 + 24);
          }
LABEL_46:
          v22 = *(unsigned __int8 *)(v27 + 8);
          if ( (_BYTE)v22 == 16 )
          {
            v100 = *(_QWORD *)(v27 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
          }
          else
          {
            v29 = v27 & 0xFFFFFFFFFFFFFFF9LL;
            if ( (unsigned int)(unsigned __int8)v22 - 17 > 1 )
            {
              if ( (_BYTE)v22 != 15 )
                v29 = 0;
              v100 = v29;
            }
            else
            {
              v100 = v29 | 2;
            }
          }
          v99 += 4;
          if ( v93 != (unsigned __int8 *)v23 )
          {
            v20 = v100;
            continue;
          }
          v5 = v93;
          v21 = *((_DWORD *)v93 + 1) & 0x7FFFFFF;
LABEL_145:
          v5 = *(unsigned __int8 **)&v5[-32 * v21];
          goto LABEL_25;
        }
      }
    }
    break;
  }
  if ( *(_BYTE *)v24 == 17 )
  {
    v22 = *(unsigned int *)(v24 + 32);
    v30 = (char **)(v24 + 24);
    if ( (unsigned int)v22 <= 0x40 )
    {
      v32 = *(_QWORD *)(v24 + 24) == 0;
    }
    else
    {
      v91 = *(_DWORD *)(v24 + 32);
      v31 = sub_C444A0(v24 + 24);
      v22 = v91;
      v32 = v91 == v31;
    }
    if ( v32 )
    {
LABEL_77:
      v23 += 4;
      v25 = v20 & 0xFFFFFFFFFFFFFFF8LL;
      v27 = v20 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v20 )
        goto LABEL_44;
      goto LABEL_78;
    }
    v33 = sub_9914A0((__int64)&v99, a3);
    v104 = v34;
    v103 = v33;
    if ( (_BYTE)v34 )
      goto LABEL_55;
    sub_C44B10((__int64)&v105, v30, v97);
    sub_C47170((__int64)&v105, v103);
    v87 = v105.m128i_i32[2];
    v105.m128i_i32[2] = 0;
    v112.m128i_i32[2] = v87;
    v112.m128i_i64[0] = v105.m128i_i64[0];
    sub_C45EE0(v95, v112.m128i_i64);
    if ( v112.m128i_i32[2] > 0x40u && v112.m128i_i64[0] )
      j_j___libc_free_0_0(v112.m128i_i64[0]);
    if ( v105.m128i_i32[2] <= 0x40u || (v52 = v105.m128i_i64[0]) == 0 )
    {
LABEL_76:
      v20 = v100;
      goto LABEL_77;
    }
LABEL_75:
    j_j___libc_free_0_0(v52);
    goto LABEL_76;
  }
  v92 = *v23;
  v53 = sub_9914A0((__int64)&v99, a3);
  v102 = v54;
  v101 = v53;
  if ( !(_BYTE)v54 )
  {
    v55 = *(_DWORD *)(*(_QWORD *)(v92 + 8) + 8LL);
    v56 = v93[1];
    v112.m128i_i64[0] = v92;
    v57 = (v56 & 8) != 0;
    v112.m128i_i32[2] = 0;
    v58 = v55 >> 8;
    v59 = v56 >> 2;
    v60 = v57 & (v56 >> 2);
    v61 = v97 - v58;
    v62 = v59 & 1;
    if ( v97 <= v58 )
      v61 = 0;
    v63 = v58 - v97;
    v64 = v97 < v58;
    v65 = 0;
    if ( v64 )
      v65 = v63;
    BYTE4(v113) = v60;
    v112.m128i_i32[3] = v61;
    LODWORD(v113) = v65;
    sub_D02480(&v105, &v112, 0);
    LODWORD(v104) = v97;
    if ( v97 > 0x40 )
      sub_C43690((__int64)&v103, (unsigned int)v101, 0);
    else
      v103 = (unsigned int)v101;
    sub_D00E00(&v112, &v105, (__int64)&v103, v57, v62);
    v66 = _mm_loadu_si128(&v112);
    LODWORD(v106) = v113;
    v105 = v66;
    BYTE4(v106) = BYTE4(v113);
    if ( v108 > 0x40 && v107 )
      j_j___libc_free_0_0(v107);
    v107 = v114;
    v67 = v115;
    LODWORD(v115) = 0;
    v108 = v67;
    if ( v110 > 0x40 && v109 )
    {
      j_j___libc_free_0_0(v109);
      v109 = *((_QWORD *)&v115 + 1);
      v110 = v116;
      v111 = v117;
      if ( (unsigned int)v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
    }
    else
    {
      v109 = *((_QWORD *)&v115 + 1);
      v110 = v116;
      v111 = v117;
    }
    if ( (unsigned int)v104 > 0x40 && v103 )
      j_j___libc_free_0_0(v103);
    sub_C45EE0(v95, &v109);
    LODWORD(v104) = v108;
    if ( v108 > 0x40 )
      sub_C43780((__int64)&v103, (const void **)&v107);
    else
      v103 = v107;
    if ( !(_BYTE)v111 )
      *(_DWORD *)(a1 + 264) &= ~4u;
    v68 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v68 )
    {
      v69 = 0;
      v70 = 56 * v68;
      do
      {
        v71 = (unsigned __int8 *)v105.m128i_i64[0];
        v72 = v69 + *(_QWORD *)(a1 + 24);
        if ( *(_QWORD *)v72 == v105.m128i_i64[0] )
        {
          v73 = v105.m128i_i64[0];
        }
        else
        {
          if ( !(unsigned __int8)sub_D033B0(*(unsigned __int8 **)v72) || !(unsigned __int8)sub_D033B0(v71) )
            goto LABEL_107;
          v73 = v105.m128i_i64[0];
          v72 = v69 + *(_QWORD *)(a1 + 24);
          v71 = *(unsigned __int8 **)v72;
        }
        if ( *((_QWORD *)v71 + 1) == *(_QWORD *)(v73 + 8) )
        {
          if ( *(_QWORD *)(v72 + 8) == v105.m128i_i64[1] && *(_DWORD *)(v72 + 16) == (_DWORD)v106
            || (*(_BYTE *)(v72 + 20) || BYTE4(v106))
            && (v22 = (unsigned int)(*(_DWORD *)(v72 + 8) + *(_DWORD *)(v72 + 12)),
                (_DWORD)v22 == v105.m128i_i32[2] + v105.m128i_i32[3])
            && *(_DWORD *)(v72 + 16) == (_DWORD)v106 )
          {
            sub_C45EE0((__int64)&v103, (__int64 *)(v72 + 24));
            v36 = *(unsigned int *)(a1 + 32);
            v37 = *(_QWORD *)(a1 + 24);
            v111 = 0;
            v38 = v37 + v69;
            v39 = v36;
            v40 = 7 * v36;
            v22 = v37 + v69 + 56;
            v41 = 8 * v40 - (v69 + 56);
            if ( v41 > 0 )
            {
              v42 = 0x6DB6DB6DB6DB6DB7LL * (v41 >> 3);
              for ( i = v37 + v69 + 56; ; i += 56 )
              {
                v44 = _mm_loadu_si128((const __m128i *)(v38 + 56));
                v45 = *(_DWORD *)(v38 + 32) <= 0x40u;
                *(_DWORD *)(v38 + 16) = *(_DWORD *)(v38 + 72);
                v46 = *(_BYTE *)(v38 + 76);
                *(__m128i *)v38 = v44;
                *(_BYTE *)(v38 + 20) = v46;
                if ( !v45 )
                {
                  v47 = *(_QWORD *)(v38 + 24);
                  if ( v47 )
                    j_j___libc_free_0_0(v47);
                }
                *(_QWORD *)(v38 + 24) = *(_QWORD *)(v38 + 80);
                v48 = *(_DWORD *)(v38 + 88);
                *(_DWORD *)(v38 + 88) = 0;
                *(_DWORD *)(v38 + 32) = v48;
                *(_QWORD *)(v38 + 40) = *(_QWORD *)(v38 + 96);
                *(_BYTE *)(v38 + 48) = *(_BYTE *)(v38 + 104);
                *(_BYTE *)(v38 + 49) = *(_BYTE *)(v38 + 105);
                v38 = i;
                if ( !--v42 )
                  break;
              }
              v39 = *(_DWORD *)(a1 + 32);
              v37 = *(_QWORD *)(a1 + 24);
            }
            v49 = (unsigned int)(v39 - 1);
            *(_DWORD *)(a1 + 32) = v49;
            v50 = v37 + 56 * v49;
            if ( *(_DWORD *)(v50 + 32) > 0x40u )
            {
              v86 = *(_QWORD *)(v50 + 24);
              if ( v86 )
                j_j___libc_free_0_0(v86);
            }
            break;
          }
        }
LABEL_107:
        v69 += 56;
      }
      while ( v69 != v70 );
    }
    v51 = v104;
    if ( (unsigned int)v104 <= 0x40 )
    {
      if ( v103 )
      {
        v114 = v103;
        v77 = _mm_loadu_si128(&v105);
        v115 = 0;
        v116 = 0;
        v113 = v106;
        LODWORD(v115) = v104;
        v112 = v77;
        goto LABEL_128;
      }
    }
    else
    {
      if ( v51 == (unsigned int)sub_C444A0((__int64)&v103) )
        goto LABEL_68;
      v88 = _mm_loadu_si128(&v105);
      v115 = 0;
      v116 = 0;
      v113 = v106;
      LODWORD(v115) = v51;
      v112 = v88;
      sub_C43780((__int64)&v114, (const void **)&v103);
LABEL_128:
      v22 = *(unsigned int *)(a1 + 32);
      v78 = *(unsigned int *)(a1 + 36);
      v79 = *(_QWORD *)(a1 + 24);
      *((_QWORD *)&v115 + 1) = v94;
      v80 = v22 + 1;
      v81 = &v112;
      LOBYTE(v116) = HIBYTE(v111);
      v82 = v22;
      if ( v22 + 1 > v78 )
      {
        v89 = a1 + 24;
        if ( v79 > (unsigned __int64)&v112 || (unsigned __int64)&v112 >= v79 + 56 * v22 )
        {
          sub_D00C80(v89, v22 + 1, v22, v78, v80, v35);
          v22 = *(unsigned int *)(a1 + 32);
          v79 = *(_QWORD *)(a1 + 24);
          v81 = &v112;
          v82 = *(_DWORD *)(a1 + 32);
        }
        else
        {
          v90 = &v112.m128i_i8[-v79];
          sub_D00C80(v89, v22 + 1, v22, v78, v80, v35);
          v79 = *(_QWORD *)(a1 + 24);
          v22 = *(unsigned int *)(a1 + 32);
          v81 = (__m128i *)&v90[v79];
          v82 = *(_DWORD *)(a1 + 32);
        }
      }
      v83 = (__m128i *)(v79 + 56 * v22);
      if ( v83 )
      {
        v84 = _mm_loadu_si128(v81);
        v83[1].m128i_i64[0] = v81[1].m128i_i64[0];
        *v83 = v84;
        v85 = v81[2].m128i_u32[0];
        v83[2].m128i_i32[0] = v85;
        if ( v85 > 0x40 )
          sub_C43780((__int64)&v83[1].m128i_i64[1], (const void **)&v81[1].m128i_i64[1]);
        else
          v83[1].m128i_i64[1] = v81[1].m128i_i64[1];
        v83[2].m128i_i64[1] = v81[2].m128i_i64[1];
        v83[3].m128i_i16[0] = v81[3].m128i_i16[0];
        v82 = *(_DWORD *)(a1 + 32);
      }
      v45 = (unsigned int)v115 <= 0x40;
      *(_DWORD *)(a1 + 32) = v82 + 1;
      if ( !v45 && v114 )
        j_j___libc_free_0_0(v114);
      if ( (unsigned int)v104 > 0x40 )
      {
LABEL_68:
        if ( v103 )
          j_j___libc_free_0_0(v103);
      }
    }
    if ( v110 > 0x40 && v109 )
      j_j___libc_free_0_0(v109);
    if ( v108 <= 0x40 )
      goto LABEL_76;
    v52 = v107;
    if ( !v107 )
      goto LABEL_76;
    goto LABEL_75;
  }
LABEL_55:
  v5 = v93;
LABEL_14:
  *(_QWORD *)a1 = v5;
  return a1;
}
