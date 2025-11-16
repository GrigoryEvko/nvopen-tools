// Function: sub_1EDC030
// Address: 0x1edc030
//
__int64 __fastcall sub_1EDC030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char v7; // cl
  int v8; // edx
  int v9; // r10d
  __int64 v10; // r15
  int v11; // r11d
  unsigned __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r13
  __int64 v15; // r14
  unsigned int v16; // edx
  __int64 v17; // r13
  __int64 v18; // r11
  __int64 v19; // rbx
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rcx
  unsigned int v24; // esi
  __int64 *v25; // rax
  __int64 v26; // r11
  unsigned __int64 v27; // r13
  __int64 v28; // r15
  __int64 *v29; // rdx
  unsigned int v30; // r10d
  __int64 *v32; // r11
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  bool v35; // r10
  _DWORD *v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rsi
  int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 *v43; // r12
  __int64 v44; // rcx
  _QWORD *v45; // rsi
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 *v50; // r11
  unsigned __int8 v51; // r10
  __int64 v52; // rbx
  signed __int64 v53; // r12
  __int64 *v54; // rax
  __int64 v55; // r9
  const __m128i *v56; // rax
  unsigned __int64 v57; // rsi
  __int64 v58; // r14
  __int64 *v59; // rdx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rcx
  __int64 v63; // rcx
  unsigned int v64; // ebx
  __int64 v65; // rcx
  unsigned int v66; // r14d
  __int64 v67; // rdx
  int v68; // eax
  __int64 v69; // r10
  __int64 v70; // rsi
  _QWORD *v71; // rcx
  _QWORD *v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rsi
  _QWORD *v75; // rdx
  _QWORD *v76; // rax
  int v77; // r8d
  unsigned int v78; // eax
  __int64 v79; // rax
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  unsigned __int64 v83; // r12
  __int64 v84; // r13
  unsigned int v85; // ebx
  __int64 *v86; // rdx
  _QWORD *v87; // r13
  unsigned int v88; // eax
  __int128 v89; // [rsp-20h] [rbp-100h]
  __int128 v90; // [rsp-20h] [rbp-100h]
  unsigned __int8 v91; // [rsp+Fh] [rbp-D1h]
  bool v92; // [rsp+10h] [rbp-D0h]
  __int64 *v93; // [rsp+10h] [rbp-D0h]
  bool v94; // [rsp+18h] [rbp-C8h]
  __int64 *v95; // [rsp+18h] [rbp-C8h]
  __int64 v96; // [rsp+18h] [rbp-C8h]
  __int64 v97; // [rsp+20h] [rbp-C0h]
  __int64 v98; // [rsp+28h] [rbp-B8h]
  __int64 *v99; // [rsp+30h] [rbp-B0h]
  __int64 v100; // [rsp+30h] [rbp-B0h]
  __int64 v101; // [rsp+30h] [rbp-B0h]
  __int64 *v102; // [rsp+38h] [rbp-A8h]
  __int64 v103; // [rsp+38h] [rbp-A8h]
  unsigned __int8 v104; // [rsp+38h] [rbp-A8h]
  __int64 *v105; // [rsp+40h] [rbp-A0h]
  __int64 v106; // [rsp+40h] [rbp-A0h]
  int v107; // [rsp+40h] [rbp-A0h]
  int v108; // [rsp+40h] [rbp-A0h]
  __int64 v109; // [rsp+48h] [rbp-98h]
  __int64 v110; // [rsp+48h] [rbp-98h]
  __int64 *v111; // [rsp+48h] [rbp-98h]
  unsigned __int8 v113; // [rsp+50h] [rbp-90h]
  __m128i v115; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v116[14]; // [rsp+70h] [rbp-70h] BYREF

  v7 = *(_BYTE *)(a2 + 26);
  v8 = *(_DWORD *)(a2 + 8);
  v9 = *(_DWORD *)(a2 + 12);
  v10 = *(_QWORD *)(a1 + 272);
  v11 = v8;
  if ( !v7 )
    v11 = *(_DWORD *)(a2 + 12);
  v12 = *(unsigned int *)(v10 + 408);
  v13 = v11 & 0x7FFFFFFF;
  v14 = v11 & 0x7FFFFFFF;
  if ( (v11 & 0x7FFFFFFFu) >= (unsigned int)v12 || (v15 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8LL * v13)) == 0 )
  {
    v66 = v13 + 1;
    if ( (unsigned int)v12 < v13 + 1 )
    {
      v69 = v66;
      if ( v66 < v12 )
      {
        *(_DWORD *)(v10 + 408) = v66;
      }
      else if ( v66 > v12 )
      {
        if ( v66 > (unsigned __int64)*(unsigned int *)(v10 + 412) )
        {
          v107 = v11;
          sub_16CD150(v10 + 400, (const void *)(v10 + 416), v66, 8, a5, a6);
          v12 = *(unsigned int *)(v10 + 408);
          v11 = v107;
          v69 = v66;
        }
        v67 = *(_QWORD *)(v10 + 400);
        v70 = *(_QWORD *)(v10 + 416);
        v71 = (_QWORD *)(v67 + 8 * v69);
        v72 = (_QWORD *)(v67 + 8 * v12);
        if ( v71 != v72 )
        {
          do
            *v72++ = v70;
          while ( v71 != v72 );
          v67 = *(_QWORD *)(v10 + 400);
        }
        *(_DWORD *)(v10 + 408) = v66;
        goto LABEL_60;
      }
    }
    v67 = *(_QWORD *)(v10 + 400);
LABEL_60:
    *(_QWORD *)(v67 + 8 * v14) = sub_1DBA290(v11);
    v15 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8 * v14);
    sub_1DBB110((_QWORD *)v10, v15);
    v8 = *(_DWORD *)(a2 + 8);
    v9 = *(_DWORD *)(a2 + 12);
    v7 = *(_BYTE *)(a2 + 26);
    v10 = *(_QWORD *)(a1 + 272);
    v12 = *(unsigned int *)(v10 + 408);
  }
  if ( !v7 )
    v9 = v8;
  v16 = v9 & 0x7FFFFFFF;
  v17 = v9 & 0x7FFFFFFF;
  v18 = 8 * v17;
  if ( (v9 & 0x7FFFFFFFu) >= (unsigned int)v12 || (v19 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8LL * v16)) == 0 )
  {
    v64 = v16 + 1;
    if ( v16 + 1 > (unsigned int)v12 )
    {
      v73 = v64;
      if ( v64 < v12 )
      {
        *(_DWORD *)(v10 + 408) = v64;
      }
      else if ( v64 > v12 )
      {
        if ( v64 > (unsigned __int64)*(unsigned int *)(v10 + 412) )
        {
          v103 = 8LL * (v9 & 0x7FFFFFFF);
          v108 = v9;
          sub_16CD150(v10 + 400, (const void *)(v10 + 416), v64, 8, a5, a6);
          v12 = *(unsigned int *)(v10 + 408);
          v18 = v103;
          v9 = v108;
          v73 = v64;
        }
        v65 = *(_QWORD *)(v10 + 400);
        v74 = *(_QWORD *)(v10 + 416);
        v75 = (_QWORD *)(v65 + 8 * v73);
        v76 = (_QWORD *)(v65 + 8 * v12);
        if ( v75 != v76 )
        {
          do
            *v76++ = v74;
          while ( v75 != v76 );
          v65 = *(_QWORD *)(v10 + 400);
        }
        *(_DWORD *)(v10 + 408) = v64;
        goto LABEL_57;
      }
    }
    v65 = *(_QWORD *)(v10 + 400);
LABEL_57:
    *(_QWORD *)(v65 + v18) = sub_1DBA290(v9);
    v19 = *(_QWORD *)(*(_QWORD *)(v10 + 400) + 8 * v17);
    sub_1DBB110((_QWORD *)v10, v19);
    v10 = *(_QWORD *)(a1 + 272);
  }
  v20 = *(_QWORD *)(v10 + 272);
  v21 = a3;
  if ( (*(_BYTE *)(a3 + 46) & 4) != 0 )
  {
    do
      v21 = *(_QWORD *)v21 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v21 + 46) & 4) != 0 );
  }
  v22 = *(_QWORD *)(v20 + 368);
  v23 = *(unsigned int *)(v20 + 384);
  if ( (_DWORD)v23 )
  {
    v24 = (v23 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v21 == *v25 )
      goto LABEL_13;
    v68 = 1;
    while ( v26 != -8 )
    {
      v77 = v68 + 1;
      v24 = (v23 - 1) & (v68 + v24);
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v21 == *v25 )
        goto LABEL_13;
      v68 = v77;
    }
  }
  v25 = (__int64 *)(v22 + 16 * v23);
LABEL_13:
  v27 = v25[1] & 0xFFFFFFFFFFFFFFF8LL;
  v28 = v27 | 4;
  v29 = (__int64 *)sub_1DB3C70((__int64 *)v19, v27 | 4);
  if ( v29 == (__int64 *)(*(_QWORD *)v19 + 24LL * *(unsigned int *)(v19 + 8)) )
    return 0;
  if ( (*(_DWORD *)((*v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v29 >> 1) & 3) > (*(_DWORD *)(v27 + 24) | 2u) )
    return 0;
  if ( *(_QWORD *)(v29[2] + 8) != v28 )
    return 0;
  v105 = v29;
  v109 = v29[2];
  v97 = v27 | 2;
  v32 = (__int64 *)sub_1DB3C70((__int64 *)v15, v27 | 2);
  if ( v32 == (__int64 *)(*(_QWORD *)v15 + 24LL * *(unsigned int *)(v15 + 8))
    || (*(_DWORD *)((*v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v32 >> 1) & 3) > (*(_DWORD *)(v27 + 24) | 1u) )
  {
    return 0;
  }
  v98 = v32[2];
  v33 = *(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v34 = v33 ? *(_QWORD *)(v33 + 16) : 0LL;
  v99 = v32;
  v102 = v105;
  v35 = sub_1EDB0A0((unsigned int *)a2, v34);
  if ( !v35 )
    return 0;
  if ( **(_WORD **)(v34 + 16) != 15 )
    return 0;
  v36 = *(_DWORD **)(v34 + 32);
  if ( (*v36 & 0xFFF00) != 0 || (v36[10] & 0xFFF00) != 0 )
    return 0;
  v37 = *(_QWORD *)(v98 + 8);
  v38 = v37 & 0xFFFFFFFFFFFFFFF8LL;
  v39 = (v37 >> 1) & 3;
  v40 = v39 ? (2LL * (v39 - 1)) | v38 : *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v94 = v35;
  v43 = (__int64 *)sub_1DB3C70((__int64 *)v19, v40);
  if ( v43 == (__int64 *)(*(_QWORD *)v19 + 24LL * *(unsigned int *)(v19 + 8)) )
    return 0;
  v44 = v109;
  if ( (*(_DWORD *)((*v43 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v43 >> 1) & 3) > (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v40 >> 1) & 3) )
    return 0;
  v110 = v43[1];
  v45 = (_QWORD *)(v110 & 0xFFFFFFFFFFFFFFF8LL);
  v46 = ((v110 >> 1) & 3) != 0
      ? v110 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v110 >> 1) & 3) - 1)) & 0xFFFFFFFFFFFFFFF8LL
      : *v45 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v46
    && (v47 = *(_QWORD *)(v46 + 16), (v106 = v47) != 0)
    && *(_QWORD *)(a3 + 24) == *(_QWORD *)(v47 + 24)
    && v102 == v43 + 3 )
  {
    v48 = *v102;
    v92 = v94;
    v95 = v99;
    *(_QWORD *)(v44 + 8) = v110;
    v115.m128i_i64[1] = v48;
    v116[0] = v44;
    *((_QWORD *)&v89 + 1) = v48;
    *(_QWORD *)&v89 = v110;
    v100 = v44;
    v115.m128i_i64[0] = v110;
    sub_1DB8610(v19, (__int64)v45, (__int64)v102, v44, v41, v42, v89, v44);
    v49 = v43[2];
    v50 = v95;
    v51 = v92;
    if ( v49 != v100 )
    {
      sub_1DB4840(v19, v100, v49);
      v51 = v92;
      v50 = v95;
    }
    if ( *(_QWORD *)(v19 + 104) )
    {
      v96 = v19;
      v52 = *(_QWORD *)(v19 + 104);
      v101 = v15;
      v93 = v50;
      v91 = v51;
      do
      {
        v56 = (const __m128i *)sub_1DB3C70((__int64 *)v52, v28);
        if ( v56 != (const __m128i *)(*(_QWORD *)v52 + 24LL * *(unsigned int *)(v52 + 8))
          && (v57 = v56->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL,
              (*(_DWORD *)(v57 + 24) | (unsigned int)(v56->m128i_i64[0] >> 1) & 3) <= (*(_DWORD *)(v27 + 24) | 2u))
          && v57 == (v56->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v115 = _mm_loadu_si128(v56);
          v116[0] = v56[1].m128i_i64[0];
          sub_1DB4410(v52, v115.m128i_i64[0], v115.m128i_i64[1], 1);
        }
        else
        {
          v58 = 0;
          v59 = (__int64 *)sub_1DB3C70((__int64 *)v52, v28);
          v62 = 3LL * *(unsigned int *)(v52 + 8);
          if ( v59 != (__int64 *)(*(_QWORD *)v52 + 24LL * *(unsigned int *)(v52 + 8)) )
          {
            v62 = *(_DWORD *)(v27 + 24) | 2u;
            if ( (*(_DWORD *)((*v59 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v59 >> 1) & 3) <= (unsigned int)v62 )
              v58 = v59[2];
          }
          v116[0] = v58;
          v115.m128i_i64[1] = v48;
          *((_QWORD *)&v90 + 1) = v48;
          *(_QWORD *)&v90 = v110;
          v115.m128i_i64[0] = v110;
          sub_1DB8610(v52, v28, (__int64)v59, v62, v60, v61, v90, v58);
          v63 = *(_QWORD *)(v98 + 8);
          if ( ((v63 >> 1) & 3) != 0 )
            v53 = v63 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v63 >> 1) & 3) - 1));
          else
            v53 = *(_QWORD *)(v63 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
          v54 = (__int64 *)sub_1DB3C70((__int64 *)v52, v53);
          v55 = 0;
          if ( v54 != (__int64 *)(*(_QWORD *)v52 + 24LL * *(unsigned int *)(v52 + 8))
            && (*(_DWORD *)((*v54 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v54 >> 1) & 3)) <= (*(_DWORD *)((v53 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v53 >> 1) & 3) )
          {
            v55 = v54[2];
          }
          if ( v55 != v58 )
            sub_1DB4840(v52, v58, v55);
        }
        v52 = *(_QWORD *)(v52 + 104);
      }
      while ( v52 );
      v15 = v101;
      v19 = v96;
      v50 = v93;
      v51 = v91;
    }
    v104 = v51;
    v111 = v50;
    v78 = sub_1E165A0(v106, *(_DWORD *)(v19 + 112), 1, 0);
    if ( v78 != -1 )
    {
      v79 = *(_QWORD *)(v106 + 32) + 40LL * v78;
      *(_BYTE *)(v79 + 3) &= ~0x40u;
    }
    sub_1E17170(a3, *(_DWORD *)(v15 + 112), *(_DWORD *)(v19 + 112), 0, *(_QWORD *)(a1 + 256));
    v30 = v104;
    if ( v28 != v111[1] )
    {
      if ( !*(_QWORD *)(v15 + 104) )
        return v30;
      v83 = v27;
      v84 = *(_QWORD *)(v15 + 104);
      v85 = v104;
      while ( 1 )
      {
        v86 = (__int64 *)sub_1DB3C70((__int64 *)v84, v97);
        if ( v86 != (__int64 *)(*(_QWORD *)v84 + 24LL * *(unsigned int *)(v84 + 8))
          && (*(_DWORD *)((*v86 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v86 >> 1) & 3) <= (*(_DWORD *)(v83 + 24)
                                                                                                 | 1u)
          && v28 == v86[1] )
        {
          break;
        }
        v84 = *(_QWORD *)(v84 + 104);
        if ( !v84 )
          return v85;
      }
      LOBYTE(v30) = v104;
    }
    v113 = v30;
    v87 = *(_QWORD **)(a1 + 272);
    v88 = sub_1DC0580(v87, v15, 0, v80, v81, v82);
    v30 = v113;
    v85 = v88;
    if ( (_BYTE)v88 )
    {
      v115.m128i_i64[0] = (__int64)v116;
      v115.m128i_i64[1] = 0x800000000LL;
      sub_1DBEB50((__int64)v87, v15, (__int64)&v115);
      if ( (_QWORD *)v115.m128i_i64[0] != v116 )
        _libc_free(v115.m128i_u64[0]);
      return v85;
    }
  }
  else
  {
    return 0;
  }
  return v30;
}
