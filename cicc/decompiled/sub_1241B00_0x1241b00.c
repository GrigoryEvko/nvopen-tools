// Function: sub_1241B00
// Address: 0x1241b00
//
__int64 __fastcall sub_1241B00(__int64 a1, __int64 *a2)
{
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  __int32 v10; // r15d
  unsigned __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // r9
  __int64 v14; // rdx
  unsigned __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rcx
  int *v18; // r8
  __int64 v19; // r9
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rdx
  _BYTE *v26; // rdi
  __int64 v27; // r14
  const __m128i *v28; // r11
  const __m128i *v29; // rcx
  unsigned __int64 v30; // r12
  __int64 v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // rax
  const __m128i *v38; // r15
  __int64 *v39; // r14
  __int64 *v40; // rcx
  const __m128i *v41; // rbx
  __int64 v42; // r13
  const __m128i **v43; // r12
  __int64 v44; // rax
  char *v45; // rax
  unsigned int v46; // r12d
  _QWORD *v48; // rax
  __m128i *v49; // rsi
  _QWORD *v50; // [rsp+8h] [rbp-248h]
  __int64 v51; // [rsp+8h] [rbp-248h]
  __int64 v52; // [rsp+20h] [rbp-230h]
  int v53; // [rsp+30h] [rbp-220h]
  __int64 v54; // [rsp+38h] [rbp-218h]
  __int64 v55; // [rsp+38h] [rbp-218h]
  __int64 v56; // [rsp+40h] [rbp-210h]
  __int64 *v57; // [rsp+48h] [rbp-208h]
  unsigned int v58; // [rsp+54h] [rbp-1FCh] BYREF
  __int64 v59; // [rsp+58h] [rbp-1F8h] BYREF
  __int64 v60; // [rsp+60h] [rbp-1F0h] BYREF
  int v61; // [rsp+68h] [rbp-1E8h] BYREF
  _QWORD *v62; // [rsp+70h] [rbp-1E0h]
  int *v63; // [rsp+78h] [rbp-1D8h]
  int *v64; // [rsp+80h] [rbp-1D0h]
  __int64 v65; // [rsp+88h] [rbp-1C8h]
  _BYTE *v66; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v67; // [rsp+98h] [rbp-1B8h]
  _BYTE v68[48]; // [rsp+A0h] [rbp-1B0h] BYREF
  _BYTE *v69; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v70; // [rsp+D8h] [rbp-178h]
  _BYTE v71[48]; // [rsp+E0h] [rbp-170h] BYREF
  char *v72; // [rsp+110h] [rbp-140h] BYREF
  __int64 v73; // [rsp+118h] [rbp-138h]
  _BYTE v74[48]; // [rsp+120h] [rbp-130h] BYREF
  __m128i v75; // [rsp+150h] [rbp-100h] BYREF
  _BYTE v76[48]; // [rsp+160h] [rbp-F0h] BYREF
  __m128i v77; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+1A0h] [rbp-B0h]
  _QWORD v79[6]; // [rsp+1A8h] [rbp-A8h] BYREF
  char *v80; // [rsp+1D8h] [rbp-78h] BYREF
  __int64 v81; // [rsp+1E0h] [rbp-70h]
  _BYTE v82[104]; // [rsp+1E8h] [rbp-68h] BYREF

  v63 = &v61;
  v64 = &v61;
  v61 = 0;
  v62 = 0;
  v65 = 0;
  while ( 1 )
  {
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in callsite")
      || (unsigned __int8)sub_120AFE0(a1, 440, "expected 'callee' in callsite")
      || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
    {
      goto LABEL_97;
    }
    v4 = *(_QWORD *)(a1 + 232);
    v5 = *(_DWORD *)(a1 + 240) == 54;
    v59 = 0;
    v58 = 0;
    v54 = v4;
    if ( v5 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    else if ( (unsigned __int8)sub_12122D0(a1, &v59, &v58) )
    {
      goto LABEL_97;
    }
    if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in callsite")
      || (unsigned __int8)sub_120AFE0(a1, 493, "expected 'clones' in callsite")
      || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
      || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in clones") )
    {
      goto LABEL_97;
    }
    v67 = 0xC00000000LL;
    v66 = v68;
    while ( 1 )
    {
      v77.m128i_i32[0] = 0;
      v6 = (__int64)&v77;
      if ( (unsigned __int8)sub_120BD00(a1, &v77) )
        goto LABEL_95;
      v9 = (unsigned int)v67;
      v10 = v77.m128i_i32[0];
      v11 = (unsigned int)v67 + 1LL;
      if ( v11 > HIDWORD(v67) )
      {
        sub_C8D5F0((__int64)&v66, v68, v11, 4u, v7, v8);
        v9 = (unsigned int)v67;
      }
      *(_DWORD *)&v66[4 * v9] = v10;
      LODWORD(v67) = v67 + 1;
      if ( *(_DWORD *)(a1 + 240) != 4 )
        break;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    v6 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in clones") )
      goto LABEL_95;
    v6 = 4;
    if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in callsite") )
      goto LABEL_95;
    v6 = 494;
    if ( (unsigned __int8)sub_120AFE0(a1, 494, "expected 'stackIds' in callsite") )
      goto LABEL_95;
    v6 = 16;
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
      goto LABEL_95;
    v6 = 12;
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in stackIds") )
      goto LABEL_95;
    v70 = 0xC00000000LL;
    v5 = *(_DWORD *)(a1 + 240) == 13;
    v69 = v71;
    if ( !v5 )
    {
      while ( 1 )
      {
        v6 = (__int64)&v77;
        v77.m128i_i64[0] = 0;
        if ( (unsigned __int8)sub_120C050(a1, v77.m128i_i64) )
          break;
        v12 = sub_9E27D0(*(_QWORD *)(a1 + 352), v77.m128i_i64[0]);
        v14 = (unsigned int)v70;
        v15 = (unsigned int)v70 + 1LL;
        if ( v15 > HIDWORD(v70) )
        {
          v53 = v12;
          sub_C8D5F0((__int64)&v69, v71, (unsigned int)v70 + 1LL, 4u, v15, v13);
          v14 = (unsigned int)v70;
          v12 = v53;
        }
        *(_DWORD *)&v69[4 * v14] = v12;
        LODWORD(v70) = v70 + 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_26;
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      }
LABEL_93:
      if ( v69 != v71 )
        _libc_free(v69, v6);
LABEL_95:
      if ( v66 != v68 )
        _libc_free(v66, v6);
LABEL_97:
      v46 = 1;
      goto LABEL_88;
    }
LABEL_26:
    v6 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in stackIds") )
      goto LABEL_93;
    if ( (v59 & 0xFFFFFFFFFFFFFFF8LL) == 0xFFFFFFFFFFFFFFF8LL )
    {
      v48 = v62;
      if ( v62 )
      {
        v18 = &v61;
        do
        {
          while ( 1 )
          {
            v17 = v48[2];
            v16 = v48[3];
            if ( *((_DWORD *)v48 + 8) >= v58 )
              break;
            v48 = (_QWORD *)v48[3];
            if ( !v16 )
              goto LABEL_108;
          }
          v18 = (int *)v48;
          v48 = (_QWORD *)v48[2];
        }
        while ( v17 );
LABEL_108:
        if ( v18 != &v61 && v58 >= v18[8] )
          goto LABEL_111;
      }
      else
      {
        v18 = &v61;
      }
      v77.m128i_i64[0] = (__int64)&v58;
      v18 = (int *)sub_1239060(&v60, (__int64)v18, (unsigned int **)&v77);
LABEL_111:
      v77.m128i_i32[0] = -252645135 * ((a2[1] - *a2) >> 3);
      v77.m128i_i64[1] = v54;
      v49 = (__m128i *)*((_QWORD *)v18 + 6);
      if ( v49 == *((__m128i **)v18 + 7) )
      {
        sub_12171B0((const __m128i **)v18 + 5, v49, &v77);
      }
      else
      {
        if ( v49 )
        {
          *v49 = _mm_loadu_si128(&v77);
          v49 = (__m128i *)*((_QWORD *)v18 + 6);
        }
        *((_QWORD *)v18 + 6) = v49 + 1;
      }
    }
    v19 = (unsigned int)v67;
    v75.m128i_i64[1] = 0xC00000000LL;
    v75.m128i_i64[0] = (__int64)v76;
    if ( (_DWORD)v67 )
      sub_1205840((__int64)&v75, (__int64)&v66, v16, v17, (__int64)v18, (unsigned int)v67);
    v20 = (unsigned int)v70;
    v73 = 0xC00000000LL;
    v72 = v74;
    if ( (_DWORD)v70 )
      sub_1205840((__int64)&v72, (__int64)&v69, v16, v17, (unsigned int)v70, v19);
    v78 = 0xC00000000LL;
    v77.m128i_i64[0] = v59;
    v77.m128i_i64[1] = (__int64)v79;
    if ( v75.m128i_i32[2] )
      sub_1205E10((__int64)&v77.m128i_i64[1], (char **)&v75, v16, v17, v20, v19);
    v21 = (unsigned int)v73;
    v81 = 0xC00000000LL;
    v80 = v82;
    if ( (_DWORD)v73 )
    {
      v21 = (__int64)&v72;
      sub_1205E10((__int64)&v80, &v72, v16, v17, v20, v19);
      v22 = a2[1];
      if ( v22 == a2[2] )
      {
LABEL_100:
        v21 = v22;
        sub_9D3840(a2, (char *)v22, v77.m128i_i64);
        goto LABEL_43;
      }
    }
    else
    {
      v22 = a2[1];
      if ( v22 == a2[2] )
        goto LABEL_100;
    }
    if ( v22 )
    {
      v23 = v77.m128i_i64[0];
      *(_QWORD *)(v22 + 16) = 0xC00000000LL;
      *(_QWORD *)v22 = v23;
      *(_QWORD *)(v22 + 8) = v22 + 24;
      v24 = (unsigned int)v78;
      if ( (_DWORD)v78 )
      {
        v21 = (__int64)&v77.m128i_i64[1];
        v51 = v22;
        sub_1205E10(v22 + 8, (char **)&v77.m128i_i64[1], v22 + 24, (unsigned int)v78, v22, v19);
        v22 = v51;
      }
      *(_QWORD *)(v22 + 80) = 0xC00000000LL;
      *(_QWORD *)(v22 + 72) = v22 + 88;
      if ( (_DWORD)v81 )
      {
        v21 = (__int64)&v80;
        sub_1205E10(v22 + 72, &v80, (unsigned int)v81, v24, v22, v19);
      }
      v22 = a2[1];
    }
    a2[1] = v22 + 136;
LABEL_43:
    if ( v80 != v82 )
      _libc_free(v80, v21);
    if ( (_QWORD *)v77.m128i_i64[1] != v79 )
      _libc_free(v77.m128i_i64[1], v21);
    if ( v72 != v74 )
      _libc_free(v72, v21);
    if ( (_BYTE *)v75.m128i_i64[0] != v76 )
      _libc_free(v75.m128i_i64[0], v21);
    v6 = 13;
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in callsite") )
      goto LABEL_93;
    if ( v69 != v71 )
      _libc_free(v69, 13);
    v26 = v66;
    if ( v66 != v68 )
      _libc_free(v66, 13);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v27 = (__int64)v63;
  v56 = a1 + 1536;
  v50 = (_QWORD *)(a1 + 1528);
  if ( v63 == &v61 )
    goto LABEL_87;
  v52 = a1;
  while ( 2 )
  {
    v77.m128i_i32[0] = *(_DWORD *)(v27 + 32);
    v28 = *(const __m128i **)(v27 + 48);
    v29 = *(const __m128i **)(v27 + 40);
    v77.m128i_i64[1] = 0;
    v78 = 0;
    v79[0] = 0;
    v30 = (char *)v28 - (char *)v29;
    if ( v28 == v29 )
    {
      v31 = 0;
    }
    else
    {
      if ( v30 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v26, v6, v25);
      v31 = sub_22077B0((char *)v28 - (char *)v29);
      v28 = *(const __m128i **)(v27 + 48);
      v29 = *(const __m128i **)(v27 + 40);
    }
    v77.m128i_i64[1] = v31;
    v78 = v31;
    v79[0] = v31 + v30;
    if ( v28 == v29 )
    {
      v33 = v31;
    }
    else
    {
      v32 = (__m128i *)v31;
      v33 = v31 + (char *)v28 - (char *)v29;
      do
      {
        if ( v32 )
          *v32 = _mm_loadu_si128(v29);
        ++v32;
        ++v29;
      }
      while ( (__m128i *)v33 != v32 );
    }
    v78 = v33;
    v34 = *(_QWORD *)(v52 + 1544);
    if ( !v34 )
    {
      v35 = v56;
      goto LABEL_74;
    }
    v35 = v56;
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v34 + 16);
        v36 = *(_QWORD *)(v34 + 24);
        if ( *(_DWORD *)(v34 + 32) >= v77.m128i_i32[0] )
          break;
        v34 = *(_QWORD *)(v34 + 24);
        if ( !v36 )
          goto LABEL_72;
      }
      v35 = v34;
      v34 = *(_QWORD *)(v34 + 16);
    }
    while ( v6 );
LABEL_72:
    if ( v35 == v56 || v77.m128i_i32[0] < *(_DWORD *)(v35 + 32) )
    {
LABEL_74:
      v6 = v35;
      v75.m128i_i64[0] = (__int64)&v77;
      v37 = sub_12395C0(v50, v35, (unsigned int **)&v75);
      v33 = v78;
      v35 = v37;
      v31 = v77.m128i_i64[1];
    }
    if ( v31 != v33 )
    {
      v55 = v27;
      v38 = (const __m128i *)v31;
      v39 = a2;
      v40 = &v75.m128i_i64[1];
      v41 = (const __m128i *)v33;
      v42 = v35;
      v43 = (const __m128i **)(v35 + 40);
      do
      {
        while ( 1 )
        {
          v44 = v38->m128i_u32[0];
          v75 = _mm_loadu_si128(v38);
          v45 = (char *)(*v39 + 136 * v44);
          v72 = v45;
          v6 = *(_QWORD *)(v42 + 48);
          if ( v6 != *(_QWORD *)(v42 + 56) )
            break;
          ++v38;
          v57 = v40;
          sub_12135D0(v43, (const __m128i *)v6, &v72, v40);
          v40 = v57;
          if ( v41 == v38 )
            goto LABEL_82;
        }
        if ( v6 )
        {
          *(_QWORD *)v6 = v45;
          *(_QWORD *)(v6 + 8) = v75.m128i_i64[1];
          v6 = *(_QWORD *)(v42 + 48);
        }
        v6 += 16;
        ++v38;
        *(_QWORD *)(v42 + 48) = v6;
      }
      while ( v41 != v38 );
LABEL_82:
      a2 = v39;
      v33 = v77.m128i_i64[1];
      v27 = v55;
    }
    if ( v33 )
    {
      v6 = v79[0] - v33;
      j_j___libc_free_0(v33, v79[0] - v33);
    }
    v26 = (_BYTE *)v27;
    v27 = sub_220EEE0(v27);
    if ( (int *)v27 != &v61 )
      continue;
    break;
  }
  a1 = v52;
LABEL_87:
  v46 = sub_120AFE0(a1, 13, "expected ')' in callsites");
LABEL_88:
  sub_1207E40(v62);
  return v46;
}
