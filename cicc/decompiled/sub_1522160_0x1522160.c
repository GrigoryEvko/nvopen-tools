// Function: sub_1522160
// Address: 0x1522160
//
__int64 *__fastcall sub_1522160(__int64 *a1, __m128i *a2, char a3)
{
  __int64 v3; // r12
  __int64 v5; // rdi
  __int64 v6; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // rdi
  __int64 *v16; // rbx
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  unsigned __int64 v19; // r8
  char v20; // dl
  __int8 v21; // al
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rbx
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 *v28; // r14
  __int64 *v29; // r12
  __int64 v30; // r15
  __int64 v31; // rdx
  _BYTE **v32; // rbx
  __int64 v33; // r13
  _BYTE *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rax
  unsigned int v38; // ecx
  __int64 v39; // r8
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r12
  volatile signed __int32 *v43; // r13
  signed __int32 v44; // edx
  signed __int32 v45; // edx
  __int64 *v46; // r15
  __int64 *v47; // r12
  __int64 v48; // r14
  __int64 v49; // rdx
  _BYTE **v50; // rbx
  __int64 v51; // r13
  _BYTE *v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rdi
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rcx
  unsigned int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // r14
  __int64 v66; // rbx
  volatile signed __int32 *v67; // r13
  signed __int32 v68; // eax
  signed __int32 v69; // eax
  _QWORD *i; // rax
  unsigned __int64 v71; // rdi
  __int64 *v72; // r9
  __int64 v73; // rcx
  unsigned int v74; // r8d
  unsigned int v75; // edi
  __int64 v76; // r10
  __int64 v77; // rdi
  __int64 v78; // r11
  __int64 v79; // rdx
  char v80; // cl
  __int64 v81; // [rsp+8h] [rbp-318h]
  __int64 v82; // [rsp+10h] [rbp-310h]
  __int64 v83; // [rsp+10h] [rbp-310h]
  unsigned int v84; // [rsp+18h] [rbp-308h]
  unsigned int v85; // [rsp+18h] [rbp-308h]
  __int64 v86; // [rsp+18h] [rbp-308h]
  __int64 v87; // [rsp+30h] [rbp-2F0h]
  unsigned int v89; // [rsp+4Ch] [rbp-2D4h] BYREF
  __m128i v90; // [rsp+50h] [rbp-2D0h] BYREF
  __int64 v91; // [rsp+60h] [rbp-2C0h]
  __int64 v92; // [rsp+68h] [rbp-2B8h]
  const char *v93; // [rsp+70h] [rbp-2B0h] BYREF
  unsigned __int64 v94; // [rsp+78h] [rbp-2A8h]
  __int64 v95; // [rsp+80h] [rbp-2A0h]
  unsigned __int64 v96; // [rsp+88h] [rbp-298h]
  __int64 v97; // [rsp+90h] [rbp-290h] BYREF
  __int64 v98; // [rsp+98h] [rbp-288h]
  unsigned __int8 *v99; // [rsp+A0h] [rbp-280h]
  unsigned __int64 v100; // [rsp+A8h] [rbp-278h]
  __int64 v101; // [rsp+B0h] [rbp-270h]
  unsigned __int64 v102; // [rsp+B8h] [rbp-268h]
  __m128i v103; // [rsp+C0h] [rbp-260h]
  __int64 v104; // [rsp+D0h] [rbp-250h]
  __int64 v105; // [rsp+D8h] [rbp-248h]
  char *v106; // [rsp+E0h] [rbp-240h] BYREF
  __int64 v107; // [rsp+E8h] [rbp-238h]
  char v108; // [rsp+F0h] [rbp-230h] BYREF
  char v109; // [rsp+F1h] [rbp-22Fh]

  v3 = (__int64)a2;
  if ( !a3 && (unsigned __int32)a2[2].m128i_i32[0] >> 1 )
  {
    v109 = 1;
    v106 = "Invalid metadata: fwd refs into function blocks";
    v108 = 3;
    sub_1514BE0(a1, (__int64)&v106);
    return a1;
  }
  v5 = a2[14].m128i_i64[1];
  v6 = *(unsigned int *)(v5 + 32);
  v87 = *(_QWORD *)(v5 + 16);
  if ( sub_15127D0(v5, 15, 0) )
  {
    v109 = 1;
    v106 = "Invalid record";
    v108 = 3;
    sub_1514BE0(a1, (__int64)&v106);
    return a1;
  }
  v97 = 0;
  v106 = &v108;
  v107 = 0x4000000000LL;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0u;
  v104 = 0;
  v105 = 0;
  sub_1516C10(&v97, 0);
  LODWORD(v8) = a2->m128i_i32[2];
  if ( !a3 || !a2[63].m128i_i8[4] || (_DWORD)v8 || byte_4F9DCA0 )
    goto LABEL_7;
  sub_1518180(&v90, a2);
  v20 = v90.m128i_i8[8] & 1;
  v21 = (2 * (v90.m128i_i8[8] & 1)) | v90.m128i_i8[8] & 0xFD;
  v90.m128i_i8[8] = v21;
  if ( v20 )
  {
    v54 = 0;
    v90.m128i_i8[8] = v21 & 0xFD;
    v55 = v90.m128i_i64[0];
    v90.m128i_i64[0] = 0;
    *a1 = v55 | 1;
    goto LABEL_75;
  }
  v8 = a2->m128i_u32[2];
  if ( !v90.m128i_i8[0] )
  {
LABEL_7:
    v89 = v8;
    while ( 1 )
    {
      v9 = sub_14ED070(a2[14].m128i_i64[1], 0);
      if ( (_DWORD)v9 == 1 )
        break;
      if ( (v9 & 0xFFFFFFFD) == 0 )
      {
        v93 = "Malformed block";
        LOWORD(v95) = 259;
        sub_1514BE0(a1, (__int64)&v93);
        goto LABEL_12;
      }
      v13 = a2[14].m128i_i64[1];
      LODWORD(v107) = 0;
      v93 = 0;
      v94 = 0;
      v14 = sub_1510D70(v13, SHIDWORD(v9), (__int64)&v106, (unsigned __int8 **)&v93);
      sub_151B070(&v90, a2->m128i_i64, (__int64 **)&v106, v14, (__int64)&v97, &v89, (__int64)v93, v94);
      if ( (v90.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = v90.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        goto LABEL_12;
      }
    }
    sub_1520420((__int64)a2, (__int64)&v97, v10, v11, v12);
    v46 = (__int64 *)a2[43].m128i_i64[0];
    if ( (__int64 *)a2[42].m128i_i64[1] != v46 )
    {
      v47 = (__int64 *)a2[42].m128i_i64[1];
      do
      {
        v48 = v47[1];
        if ( v48 )
        {
          if ( *(_BYTE *)v48 == 4 )
          {
            v49 = 8LL * *(unsigned int *)(v48 + 8);
            v50 = (_BYTE **)(v48 - v49);
            if ( v48 != v48 - v49 )
            {
              v51 = *v47;
              do
              {
                v52 = *v50;
                if ( *v50 && *v52 == 17 )
                  sub_1630830(v52, 5, v51);
                ++v50;
              }
              while ( (_BYTE **)v48 != v50 );
            }
          }
        }
        v47 += 2;
      }
      while ( v46 != v47 );
      v3 = (__int64)a2;
      v53 = a2[42].m128i_i64[1];
      if ( v53 != a2[43].m128i_i64[0] )
        a2[43].m128i_i64[0] = v53;
    }
    sub_15151E0(v3);
    *a1 = 1;
    goto LABEL_12;
  }
  v22 = (a2[40].m128i_i64[0] - a2[39].m128i_i64[1]) >> 4;
  v84 = v22 + ((a2[41].m128i_i64[1] - a2[41].m128i_i64[0]) >> 3);
  v23 = v84;
  if ( v84 < v8 )
  {
    v22 = a2->m128i_i64[0];
    v23 = a2->m128i_i64[0] + 8LL * v84;
    v24 = v23;
    v25 = a2->m128i_i64[0] + 8 * v8;
    if ( v25 != v23 )
    {
      do
      {
        v26 = *(_QWORD *)(v25 - 8);
        v25 -= 8;
        if ( v26 )
          sub_161E7C0(v25);
      }
      while ( v24 != v25 );
    }
LABEL_30:
    *(_DWORD *)(v3 + 8) = v84;
    goto LABEL_31;
  }
  if ( v84 > v8 )
  {
    if ( v84 > (unsigned __int64)a2->m128i_u32[3] )
    {
      sub_1516630((__int64)a2, v84);
      v23 = v84;
    }
    v22 = a2->m128i_i64[0];
    v23 = a2->m128i_i64[0] + 8 * v23;
    for ( i = (_QWORD *)(a2->m128i_i64[0] + 8LL * a2->m128i_u32[2]); (_QWORD *)v23 != i; ++i )
    {
      if ( i )
        *i = 0;
    }
    goto LABEL_30;
  }
LABEL_31:
  v27 = (__int64)&v97;
  sub_1520420(v3, (__int64)&v97, v23, v22, v19);
  v28 = *(__int64 **)(v3 + 680);
  if ( v28 != *(__int64 **)(v3 + 688) )
  {
    v85 = v6;
    v82 = v3;
    v29 = *(__int64 **)(v3 + 688);
    do
    {
      v30 = v28[1];
      if ( v30 )
      {
        if ( *(_BYTE *)v30 == 4 )
        {
          v31 = 8LL * *(unsigned int *)(v30 + 8);
          v32 = (_BYTE **)(v30 - v31);
          if ( v30 != v30 - v31 )
          {
            v33 = *v28;
            do
            {
              v34 = *v32;
              if ( *v32 && *v34 == 17 )
              {
                v27 = 5;
                sub_1630830(v34, 5, v33);
              }
              ++v32;
            }
            while ( (_BYTE **)v30 != v32 );
          }
        }
      }
      v28 += 2;
    }
    while ( v29 != v28 );
    v6 = v85;
    v3 = v82;
  }
  v35 = *(_QWORD *)(v3 + 680);
  if ( v35 != *(_QWORD *)(v3 + 688) )
    *(_QWORD *)(v3 + 688) = v35;
  sub_15151E0(v3);
  v36 = *(_QWORD *)(v3 + 232);
  v37 = *(unsigned int *)(v36 + 72);
  if ( (_DWORD)v37 )
  {
    v38 = *(_DWORD *)(v36 + 32);
    if ( v38 > 0x1F )
    {
      *(_DWORD *)(v36 + 32) = 32;
      *(_QWORD *)(v36 + 24) >>= (unsigned __int8)v38 - 32;
    }
    else
    {
      *(_DWORD *)(v36 + 32) = 0;
    }
    v39 = *(_QWORD *)(v36 + 40);
    v40 = *(_QWORD *)(v36 + 48);
    v41 = *(_QWORD *)(v36 + 64) + 32 * v37 - 32;
    *(_DWORD *)(v36 + 36) = *(_DWORD *)v41;
    v86 = *(_QWORD *)(v36 + 56);
    *(_QWORD *)(v36 + 40) = *(_QWORD *)(v41 + 8);
    *(_QWORD *)(v36 + 48) = *(_QWORD *)(v41 + 16);
    *(_QWORD *)(v36 + 56) = *(_QWORD *)(v41 + 24);
    *(_QWORD *)(v41 + 8) = 0;
    *(_QWORD *)(v41 + 16) = 0;
    *(_QWORD *)(v41 + 24) = 0;
    if ( v39 != v40 )
    {
      v81 = v3;
      v42 = v39;
      v83 = v39;
      do
      {
        v43 = *(volatile signed __int32 **)(v42 + 8);
        if ( v43 )
        {
          if ( &_pthread_key_create )
          {
            v44 = _InterlockedExchangeAdd(v43 + 2, 0xFFFFFFFF);
          }
          else
          {
            v44 = *((_DWORD *)v43 + 2);
            *((_DWORD *)v43 + 2) = v44 - 1;
          }
          if ( v44 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v43 + 16LL))(v43);
            if ( &_pthread_key_create )
            {
              v45 = _InterlockedExchangeAdd(v43 + 3, 0xFFFFFFFF);
            }
            else
            {
              v45 = *((_DWORD *)v43 + 3);
              *((_DWORD *)v43 + 3) = v45 - 1;
            }
            if ( v45 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v43 + 24LL))(v43);
          }
        }
        v42 += 16;
      }
      while ( v40 != v42 );
      v39 = v83;
      v3 = v81;
    }
    if ( v39 )
      j_j___libc_free_0(v39, v86 - v39);
    v62 = (unsigned int)(*(_DWORD *)(v36 + 72) - 1);
    *(_DWORD *)(v36 + 72) = v62;
    v63 = *(_QWORD *)(v36 + 64) + 32 * v62;
    v27 = *(_QWORD *)(v63 + 16);
    v56 = v63;
    v64 = *(_QWORD *)(v63 + 8);
    if ( v27 != v64 )
    {
      v65 = v56;
      v66 = v64;
      do
      {
        v67 = *(volatile signed __int32 **)(v66 + 8);
        if ( v67 )
        {
          if ( &_pthread_key_create )
          {
            v68 = _InterlockedExchangeAdd(v67 + 2, 0xFFFFFFFF);
          }
          else
          {
            v68 = *((_DWORD *)v67 + 2);
            *((_DWORD *)v67 + 2) = v68 - 1;
          }
          if ( v68 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v67 + 16LL))(v67);
            if ( &_pthread_key_create )
            {
              v69 = _InterlockedExchangeAdd(v67 + 3, 0xFFFFFFFF);
            }
            else
            {
              v69 = *((_DWORD *)v67 + 3);
              *((_DWORD *)v67 + 3) = v69 - 1;
            }
            if ( v69 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v67 + 24LL))(v67);
          }
        }
        v66 += 16;
      }
      while ( v27 != v66 );
      v56 = v65;
    }
    v57 = *(_QWORD *)(v56 + 8);
    if ( v57 )
    {
      v27 = *(_QWORD *)(v56 + 24) - v57;
      j_j___libc_free_0(v57, v27);
    }
    v36 = *(_QWORD *)(v3 + 232);
  }
  *(_DWORD *)(v36 + 32) = 0;
  v58 = 8 * v87 - v6;
  v59 = (v58 >> 3) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v36 + 16) = v59;
  v60 = v58 & 0x3F;
  if ( !v60 )
    goto LABEL_82;
  v71 = *(_QWORD *)(v36 + 8);
  if ( v59 >= v71 )
    goto LABEL_124;
  v27 = v59 + 8;
  v72 = (__int64 *)(v59 + *(_QWORD *)v36);
  if ( v71 < v59 + 8 )
  {
    v75 = v71 - v59;
    *(_QWORD *)(v36 + 24) = 0;
    v76 = v75;
    v74 = 8 * v75;
    v27 = v75 + v59;
    if ( v75 )
    {
      v77 = 0;
      v78 = 0;
      do
      {
        v79 = *((unsigned __int8 *)v72 + v77);
        v80 = 8 * v77++;
        v78 |= v79 << v80;
        *(_QWORD *)(v36 + 24) = v78;
      }
      while ( v77 != v76 );
      *(_QWORD *)(v36 + 16) = v27;
      *(_DWORD *)(v36 + 32) = v74;
      if ( v60 <= v74 )
        goto LABEL_117;
    }
    else
    {
      *(_QWORD *)(v36 + 16) = v27;
    }
LABEL_124:
    sub_16BD130("Unexpected end of file", 1);
  }
  v73 = *v72;
  *(_QWORD *)(v36 + 16) = v27;
  v74 = 64;
  *(_QWORD *)(v36 + 24) = v73;
LABEL_117:
  *(_QWORD *)(v36 + 24) >>= v60;
  *(_DWORD *)(v36 + 32) = v74 - v60;
LABEL_82:
  if ( (unsigned __int8)sub_14ED8F0(*(_QWORD *)(v3 + 232)) )
  {
    v93 = "Invalid record";
    v27 = (__int64)&v93;
    LOWORD(v95) = 259;
    sub_1514BE0(a1, (__int64)&v93);
  }
  else
  {
    *a1 = 1;
  }
  if ( (v90.m128i_i8[8] & 2) != 0 )
    sub_1517600(&v90, v27, v61);
  if ( (v90.m128i_i8[8] & 1) == 0 )
    goto LABEL_12;
  v54 = v90.m128i_i64[0];
LABEL_75:
  if ( v54 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v54 + 8LL))(v54);
LABEL_12:
  v90 = v103;
  v91 = v104;
  v92 = v105;
  v93 = (const char *)v99;
  v94 = v100;
  v95 = v101;
  v96 = v102;
  sub_1514A90((__int64 *)&v93, v90.m128i_i64);
  v15 = v97;
  if ( v97 )
  {
    v16 = (__int64 *)v102;
    v17 = v105 + 8;
    if ( v105 + 8 > v102 )
    {
      do
      {
        v18 = *v16++;
        j_j___libc_free_0(v18, 512);
      }
      while ( v17 > (unsigned __int64)v16 );
      v15 = v97;
    }
    j_j___libc_free_0(v15, 8 * v98);
  }
  if ( v106 != &v108 )
    _libc_free((unsigned __int64)v106);
  return a1;
}
