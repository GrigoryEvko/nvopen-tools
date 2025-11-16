// Function: sub_34A0610
// Address: 0x34a0610
//
_QWORD *__fastcall sub_34A0610(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __m128i *v4; // r12
  __m128i *v5; // rbx
  __m128i *v6; // r15
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i *v14; // r14
  __m128i *v15; // rdx
  __int64 v16; // rbx
  bool v17; // zf
  bool v18; // al
  char v19; // di
  unsigned __int64 v20; // rcx
  int v21; // eax
  int *v22; // rdx
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __m128i *v25; // r13
  _QWORD *v26; // r15
  _QWORD *v27; // r10
  int v28; // esi
  unsigned int v29; // edi
  int *v30; // rdx
  int v31; // r11d
  _DWORD *v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // r14
  char v39; // cl
  unsigned int v40; // eax
  int v41; // ecx
  unsigned int v42; // esi
  __int64 v43; // r12
  __int64 v44; // rdx
  char *v45; // rbx
  __int64 v46; // rax
  __int64 v47; // r14
  unsigned __int64 v48; // rdx
  char *v49; // rsi
  __int64 v50; // rdi
  __int64 v51; // rdx
  char *v52; // rax
  unsigned __int64 v53; // rdx
  unsigned int v54; // edx
  int v55; // edi
  unsigned int v56; // r10d
  unsigned int *v57; // rbx
  __int64 v58; // rcx
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rax
  bool v61; // cf
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // r12
  __int64 v64; // rax
  __m128i *v65; // r12
  __int64 v66; // rbx
  __m128i *v67; // rax
  unsigned __int64 v68; // r12
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  int v72; // r12d
  unsigned int *v73; // rbx
  __int64 v74; // rdx
  char v75; // al
  char v76; // al
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rdi
  unsigned __int64 v79; // rdi
  __int64 v80; // rcx
  __int64 v81; // rdi
  __int64 v82; // [rsp+0h] [rbp-C0h]
  __int64 v83; // [rsp+8h] [rbp-B8h]
  __int64 v84; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v85; // [rsp+18h] [rbp-A8h]
  __int64 v86; // [rsp+28h] [rbp-98h]
  _DWORD *v87; // [rsp+28h] [rbp-98h]
  __m128i *v89; // [rsp+38h] [rbp-88h]
  unsigned __int64 v90; // [rsp+40h] [rbp-80h]
  __int64 v91; // [rsp+48h] [rbp-78h]
  int *v92; // [rsp+48h] [rbp-78h]
  __m128i *v93; // [rsp+48h] [rbp-78h]
  __m128i *v94; // [rsp+48h] [rbp-78h]
  __int64 m128i_i64; // [rsp+58h] [rbp-68h]
  int *v97; // [rsp+58h] [rbp-68h]
  unsigned int v98; // [rsp+64h] [rbp-5Ch] BYREF
  unsigned int *v99; // [rsp+68h] [rbp-58h] BYREF
  int *v100; // [rsp+70h] [rbp-50h] BYREF
  __int64 v101; // [rsp+78h] [rbp-48h]
  int v102; // [rsp+80h] [rbp-40h] BYREF
  char v103; // [rsp+84h] [rbp-3Ch] BYREF

  v4 = (__m128i *)(a2 + 1);
  v5 = (__m128i *)a2[2];
  if ( !v5 )
    goto LABEL_13;
  v6 = (__m128i *)(a2 + 1);
  do
  {
    while ( 1 )
    {
      v8 = sub_34A0190((__int64)v5[2].m128i_i64, a3);
      v9 = v5[1].m128i_i64[0];
      v10 = v5[1].m128i_i64[1];
      if ( v8 )
        break;
      v6 = v5;
      v5 = (__m128i *)v5[1].m128i_i64[0];
      if ( !v9 )
        goto LABEL_6;
    }
    v5 = (__m128i *)v5[1].m128i_i64[1];
  }
  while ( v10 );
LABEL_6:
  if ( v4 == v6 )
  {
LABEL_13:
    v6 = (__m128i *)sub_22077B0(0x1C0u);
    m128i_i64 = (__int64)v6[2].m128i_i64;
    sub_349DE60(v6 + 2, a3);
    v17 = a2[5] == 0;
    v6[26].m128i_i64[0] = (__int64)v6[27].m128i_i64;
    v6[26].m128i_i64[1] = 0x200000000LL;
    if ( v17
      || (v16 = 0, v93 = (__m128i *)a2[4], v75 = sub_34A0190((__int64)v93[2].m128i_i64, m128i_i64), v15 = v93, !v75) )
    {
      v16 = sub_34A0330((__int64)a2, m128i_i64);
    }
  }
  else
  {
    v11 = (__int64)v6[2].m128i_i64;
    if ( !(unsigned __int8)sub_34A0190(a3, (__int64)v6[2].m128i_i64) )
      goto LABEL_21;
    v14 = (__m128i *)sub_22077B0(0x1C0u);
    m128i_i64 = (__int64)v14[2].m128i_i64;
    sub_349DE60(v14 + 2, a3);
    v14[26].m128i_i64[1] = 0x200000000LL;
    v14[26].m128i_i64[0] = (__int64)v14[27].m128i_i64;
    if ( (unsigned __int8)sub_34A0190((__int64)v14[2].m128i_i64, (__int64)v6[2].m128i_i64) )
    {
      if ( (__m128i *)a2[3] == v6 )
      {
LABEL_12:
        v16 = (__int64)v6;
        v15 = v6;
        v6 = v14;
LABEL_16:
        v18 = v16 != 0;
        goto LABEL_17;
      }
      v91 = sub_220EF80((__int64)v6);
      if ( (unsigned __int8)sub_34A0190(v91 + 32, m128i_i64) )
      {
        v15 = (__m128i *)v91;
        if ( *(_QWORD *)(v91 + 24) )
          goto LABEL_12;
LABEL_160:
        v6 = v14;
        v18 = 0;
LABEL_17:
        if ( v4 == v15 || v18 )
        {
          v19 = 1;
LABEL_20:
          sub_220F040(v19, (__int64)v6, v15, v4);
          ++a2[5];
          goto LABEL_21;
        }
        v11 = (__int64)v15[2].m128i_i64;
LABEL_119:
        v94 = v15;
        v76 = sub_34A0190(m128i_i64, v11);
        v15 = v94;
        v19 = v76;
        goto LABEL_20;
      }
    }
    else
    {
      if ( !(unsigned __int8)sub_34A0190((__int64)v6[2].m128i_i64, m128i_i64) )
      {
        v16 = (__int64)v6;
        v6 = v14;
        goto LABEL_121;
      }
      if ( (__m128i *)a2[4] == v6 )
      {
        v15 = v6;
        v6 = v14;
        goto LABEL_119;
      }
      v16 = sub_220EEE0((__int64)v6);
      if ( (unsigned __int8)sub_34A0190(m128i_i64, v16 + 32) )
      {
        if ( !v6[1].m128i_i64[1] )
        {
          v15 = v6;
          goto LABEL_160;
        }
        v6 = v14;
        v15 = (__m128i *)v16;
        goto LABEL_15;
      }
    }
    v6 = v14;
    v16 = sub_34A0330((__int64)a2, m128i_i64);
  }
LABEL_15:
  if ( v15 )
    goto LABEL_16;
LABEL_121:
  v77 = v6[23].m128i_u64[0];
  if ( (__m128i *)v77 != &v6[24] )
    _libc_free(v77);
  v78 = v6[6].m128i_u64[0];
  if ( (__m128i *)v78 != &v6[7] )
    _libc_free(v78);
  v79 = (unsigned __int64)v6;
  v6 = (__m128i *)v16;
  j_j___libc_free_0(v79);
LABEL_21:
  v20 = v6[26].m128i_u32[2];
  if ( (_DWORD)v20 )
  {
    *a1 = a1 + 2;
    a1[1] = 0x200000000LL;
    v74 = v6[26].m128i_u32[2];
    if ( (_DWORD)v74 )
      sub_349DD80((__int64)a1, (__int64)v6[26].m128i_i64, v74, v20, v12, v13);
    return a1;
  }
  v100 = &v102;
  v101 = 0x400000000LL;
  v21 = *(_DWORD *)(a3 + 56);
  if ( !v21 )
  {
    v43 = *(_QWORD *)(a3 + 64);
    v44 = 32LL * *(unsigned int *)(a3 + 72);
    v45 = (char *)(v43 + v44);
    if ( v43 == v43 + v44 )
    {
      v49 = (char *)(v43 + v44);
    }
    else
    {
      do
      {
        while ( *(_DWORD *)v43 != 1 )
        {
          v43 += 32;
          if ( v45 == (char *)v43 )
            goto LABEL_46;
        }
        v46 = (unsigned int)v101;
        v47 = *(_QWORD *)(v43 + 8);
        v48 = (unsigned int)v101 + 1LL;
        if ( v48 > HIDWORD(v101) )
        {
          sub_C8D5F0((__int64)&v100, &v102, v48, 4u, v12, v13);
          v46 = (unsigned int)v101;
        }
        v43 += 32;
        v100[v46] = v47;
        LODWORD(v101) = v101 + 1;
      }
      while ( v45 != (char *)v43 );
LABEL_46:
      v45 = *(char **)(a3 + 64);
      v44 = 32LL * *(unsigned int *)(a3 + 72);
      v49 = &v45[v44];
    }
    v50 = v44 >> 5;
    v51 = v44 >> 7;
    if ( v51 )
    {
      v52 = v45;
      while ( *(_DWORD *)v52 != 2 )
      {
        if ( *((_DWORD *)v52 + 8) == 2 )
        {
          v52 += 32;
          break;
        }
        if ( *((_DWORD *)v52 + 16) == 2 )
        {
          v52 += 64;
          break;
        }
        if ( *((_DWORD *)v52 + 24) == 2 )
        {
          v52 += 96;
          break;
        }
        v52 += 128;
        if ( &v45[128 * v51] == v52 )
        {
          v80 = (v49 - v52) >> 5;
          goto LABEL_132;
        }
      }
      if ( v52 == v49 )
        goto LABEL_60;
LABEL_137:
      sub_9C8C60((__int64)&v100, 0x40000000);
      v45 = *(char **)(a3 + 64);
      v81 = 32LL * *(unsigned int *)(a3 + 72);
      v49 = &v45[v81];
      v51 = v81 >> 7;
      v50 = v81 >> 5;
LABEL_138:
      if ( v51 )
      {
LABEL_60:
        while ( *(_DWORD *)v45 != 4 )
        {
          if ( *((_DWORD *)v45 + 8) == 4 )
          {
            v45 += 32;
            goto LABEL_61;
          }
          if ( *((_DWORD *)v45 + 16) == 4 )
          {
            v45 += 64;
            goto LABEL_61;
          }
          if ( *((_DWORD *)v45 + 24) == 4 )
          {
            v45 += 96;
            goto LABEL_61;
          }
          v45 += 128;
          if ( !--v51 )
          {
            v50 = (v49 - v45) >> 5;
            goto LABEL_139;
          }
        }
        goto LABEL_61;
      }
LABEL_139:
      if ( v50 != 2 )
      {
        if ( v50 != 3 )
        {
          if ( v50 != 1 )
            goto LABEL_63;
          goto LABEL_142;
        }
        if ( *(_DWORD *)v45 == 4 )
          goto LABEL_61;
        v45 += 32;
      }
      if ( *(_DWORD *)v45 == 4 )
        goto LABEL_61;
      v45 += 32;
LABEL_142:
      if ( *(_DWORD *)v45 == 4 )
      {
LABEL_61:
        if ( v45 != v49 )
          sub_9C8C60((__int64)&v100, 1073741826);
      }
LABEL_63:
      v20 = HIDWORD(v101);
      v53 = (unsigned int)v101 + 1LL;
      if ( v53 > HIDWORD(v101) )
        sub_C8D5F0((__int64)&v100, &v102, v53, 4u, v12, v13);
      v22 = &v100[(unsigned int)v101];
      goto LABEL_25;
    }
    v80 = v50;
    v52 = v45;
LABEL_132:
    if ( v80 != 2 )
    {
      if ( v80 != 3 )
      {
        if ( v80 != 1 )
          goto LABEL_138;
        goto LABEL_135;
      }
      if ( *(_DWORD *)v52 == 2 )
        goto LABEL_136;
      v52 += 32;
    }
    if ( *(_DWORD *)v52 == 2 )
      goto LABEL_136;
    v52 += 32;
LABEL_135:
    if ( *(_DWORD *)v52 != 2 )
      goto LABEL_138;
LABEL_136:
    if ( v52 == v49 )
      goto LABEL_138;
    goto LABEL_137;
  }
  v22 = &v102;
  if ( v21 != 1 )
  {
    v102 = 1073741825;
    v22 = (int *)&v103;
    LODWORD(v101) = 1;
  }
LABEL_25:
  *v22 = 0;
  v23 = (__int64)v100;
  v83 = (__int64)(a2 + 6);
  v84 = (__int64)v6[26].m128i_i64;
  LODWORD(v101) = v101 + 1;
  v92 = &v100[(unsigned int)v101];
  v97 = v100;
  if ( v92 == v100 )
    goto LABEL_100;
  v24 = a2;
  v25 = v6;
  v26 = v24;
  do
  {
    while ( 1 )
    {
      v39 = *((_BYTE *)v26 + 56);
      v40 = *v97;
      v98 = *v97;
      v41 = v39 & 1;
      if ( v41 )
      {
        v27 = v26 + 8;
        v28 = 3;
      }
      else
      {
        v42 = *((_DWORD *)v26 + 18);
        v27 = (_QWORD *)v26[8];
        if ( !v42 )
        {
          v54 = *((_DWORD *)v26 + 14);
          ++v26[6];
          v99 = 0;
          v55 = (v54 >> 1) + 1;
LABEL_67:
          v56 = 3 * v42;
          goto LABEL_68;
        }
        v28 = v42 - 1;
      }
      v29 = v28 & (37 * v40);
      v30 = (int *)&v27[4 * v29];
      v31 = *v30;
      if ( v40 == *v30 )
      {
LABEL_29:
        v32 = v30 + 2;
        v33 = -1431655765 * (unsigned int)((__int64)(*((_QWORD *)v30 + 2) - *((_QWORD *)v30 + 1)) >> 7);
        goto LABEL_30;
      }
      v72 = 1;
      v73 = 0;
      while ( v31 != -1 )
      {
        if ( !v73 && v31 == -2 )
          v73 = (unsigned int *)v30;
        v12 = (unsigned int)(v72 + 1);
        v29 = v28 & (v72 + v29);
        v30 = (int *)&v27[4 * v29];
        v31 = *v30;
        if ( v40 == *v30 )
          goto LABEL_29;
        ++v72;
      }
      v56 = 12;
      v42 = 4;
      if ( !v73 )
        v73 = (unsigned int *)v30;
      v54 = *((_DWORD *)v26 + 14);
      ++v26[6];
      v99 = v73;
      v55 = (v54 >> 1) + 1;
      if ( !(_BYTE)v41 )
      {
        v42 = *((_DWORD *)v26 + 18);
        goto LABEL_67;
      }
LABEL_68:
      if ( 4 * v55 >= v56 )
      {
        v42 *= 2;
LABEL_106:
        sub_349F4F0(v83, v42);
        sub_349D630(v83, (int *)&v98, &v99);
        v40 = v98;
        v54 = *((_DWORD *)v26 + 14);
        goto LABEL_70;
      }
      if ( v42 - *((_DWORD *)v26 + 15) - v55 <= v42 >> 3 )
        goto LABEL_106;
LABEL_70:
      v57 = v99;
      *((_DWORD *)v26 + 14) = (2 * (v54 >> 1) + 2) | v54 & 1;
      if ( *v57 != -1 )
        --*((_DWORD *)v26 + 15);
      *v57 = v40;
      v33 = 0;
      v40 = v98;
      v32 = v57 + 2;
      *(_QWORD *)v32 = 0;
      *((_QWORD *)v32 + 1) = 0;
      *((_QWORD *)v32 + 2) = 0;
LABEL_30:
      v34 = v33;
      v35 = v40;
      v36 = v25[26].m128i_u32[2];
      v20 = v25[26].m128i_u32[3];
      v37 = (v34 << 32) | v35;
      if ( v36 + 1 > v20 )
      {
        sub_C8D5F0(v84, &v25[27], v36 + 1, 8u, v12, v13);
        v36 = v25[26].m128i_u32[2];
      }
      v23 = v25[26].m128i_i64[0];
      *(_QWORD *)(v23 + 8 * v36) = v37;
      ++v25[26].m128i_i32[2];
      v38 = *((_QWORD *)v32 + 1);
      if ( v38 == *((_QWORD *)v32 + 2) )
        break;
      if ( v38 )
      {
        sub_349DE60(*((__m128i **)v32 + 1), a3);
        v38 = *((_QWORD *)v32 + 1);
      }
      ++v97;
      *((_QWORD *)v32 + 1) = v38 + 384;
      if ( v92 == v97 )
        goto LABEL_99;
    }
    v58 = v38 - *(_QWORD *)v32;
    v90 = *(_QWORD *)v32;
    v59 = 0xAAAAAAAAAAAAAAABLL * (v58 >> 7);
    if ( v59 == 0x55555555555555LL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v23 = 1;
    v60 = 1;
    if ( v59 )
      v60 = 0xAAAAAAAAAAAAAAABLL * (v58 >> 7);
    v61 = __CFADD__(v59, v60);
    v62 = v59 + v60;
    if ( v61 )
    {
      v63 = 0x7FFFFFFFFFFFFF80LL;
LABEL_81:
      v82 = v38 - *(_QWORD *)v32;
      v64 = sub_22077B0(v63);
      v58 = v82;
      v89 = (__m128i *)v64;
      v85 = v64 + v63;
      v86 = v64 + 384;
      goto LABEL_82;
    }
    if ( v62 )
    {
      if ( v62 > 0x55555555555555LL )
        v62 = 0x55555555555555LL;
      v63 = 384 * v62;
      goto LABEL_81;
    }
    v86 = 384;
    v85 = 0;
    v89 = 0;
LABEL_82:
    v20 = (unsigned __int64)v89->m128i_u64 + v58;
    if ( v20 )
      sub_349DE60((__m128i *)v20, a3);
    if ( v38 != v90 )
    {
      v87 = v32;
      v65 = v89;
      v66 = v90;
      while ( 1 )
      {
        if ( v65 )
          sub_349DE60(v65, v66);
        v66 += 384;
        v23 = (__int64)v65[24].m128i_i64;
        if ( v38 == v66 )
          break;
        v65 += 24;
      }
      v67 = v65 + 48;
      v32 = v87;
      v68 = v90;
      v86 = (__int64)v67;
      do
      {
        v69 = *(_QWORD *)(v68 + 336);
        if ( v69 != v68 + 352 )
          _libc_free(v69);
        v70 = *(_QWORD *)(v68 + 64);
        if ( v70 != v68 + 80 )
          _libc_free(v70);
        v68 += 384LL;
      }
      while ( v38 != v68 );
    }
    if ( v90 )
      j_j___libc_free_0(v90);
    ++v97;
    *(_QWORD *)v32 = v89;
    *((_QWORD *)v32 + 1) = v86;
    *((_QWORD *)v32 + 2) = v85;
  }
  while ( v92 != v97 );
LABEL_99:
  v6 = v25;
LABEL_100:
  *a1 = a1 + 2;
  a1[1] = 0x200000000LL;
  if ( v6[26].m128i_i32[2] )
    sub_349DD80((__int64)a1, (__int64)v6[26].m128i_i64, v23, v20, v12, v13);
  if ( v100 != &v102 )
    _libc_free((unsigned __int64)v100);
  return a1;
}
