// Function: sub_BB98D0
// Address: 0xbb98d0
//
__int64 __fastcall sub_BB98D0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // bl
  unsigned int v4; // r12d
  _QWORD *v6; // rbx
  __int64 *v7; // rdi
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r12
  void *v13; // rax
  __int64 *v14; // rcx
  _QWORD *i; // r14
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdx
  _QWORD *v20; // r8
  __int64 v21; // r13
  _QWORD *v22; // rbx
  __int64 v23; // r9
  _QWORD *v24; // r13
  _QWORD *v25; // r8
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rcx
  size_t v28; // rdx
  int v29; // eax
  _QWORD *v30; // r8
  _BYTE *v31; // rax
  __int64 v32; // rdx
  unsigned __int8 (__fastcall *v33)(__int64, const char *, __int64, __m128i *, __int64); // r15
  _BYTE *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __m128i *v37; // rax
  __int64 v38; // rsi
  const char *(__fastcall *v39)(__int64, __int64); // rax
  pthread_rwlock_t *v40; // rax
  __int64 v41; // rax
  __m128i *v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rdx
  const char *v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rax
  __m128i *v48; // rdx
  __int64 v49; // r12
  __m128i si128; // xmm0
  const char *(__fastcall *v51)(__int64, __int64); // rax
  pthread_rwlock_t *v52; // rax
  __int64 v53; // rax
  const char *v54; // rsi
  size_t v55; // r13
  void *v56; // rdi
  unsigned __int64 v57; // rax
  __int64 v58; // rax
  size_t v59; // rdx
  _BYTE *v60; // rdi
  const void *v61; // rsi
  _BYTE *v62; // rax
  size_t v63; // r13
  __int64 *v64; // rax
  __int64 *v65; // rbx
  char v66; // al
  unsigned __int64 v67; // r15
  __int64 *v68; // r8
  _QWORD *v69; // rax
  __int64 *v70; // rdx
  size_t v71; // r13
  void *v72; // rax
  _QWORD *v73; // rsi
  unsigned __int64 v74; // r10
  _QWORD *v75; // rcx
  unsigned __int64 v76; // rdx
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  size_t v82; // rdx
  _QWORD *v83; // [rsp+0h] [rbp-C0h]
  __int64 v84; // [rsp+8h] [rbp-B8h]
  _QWORD *v85; // [rsp+10h] [rbp-B0h]
  __int64 *v86; // [rsp+10h] [rbp-B0h]
  __int64 v89; // [rsp+28h] [rbp-98h]
  __m128i *v90; // [rsp+30h] [rbp-90h]
  __m128i v91; // [rsp+40h] [rbp-80h] BYREF
  __int64 v92[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v93[2]; // [rsp+60h] [rbp-60h] BYREF
  _OWORD *v94; // [rsp+70h] [rbp-50h] BYREF
  __int64 v95; // [rsp+78h] [rbp-48h]
  _OWORD v96[4]; // [rsp+80h] [rbp-40h] BYREF

  v2 = sub_B2BE50(a2);
  v89 = sub_B6F960(v2);
  if ( byte_4F822A0 || !(unsigned int)sub_2207590(&byte_4F822A0) )
  {
    v3 = 0;
    if ( !qword_4F822D8 )
      goto LABEL_3;
    goto LABEL_24;
  }
  qword_4F822C8 = 1;
  qword_4F822D0 = 0;
  v6 = (_QWORD *)qword_4F82348[8];
  qword_4F822D8 = 0;
  qword_4F822C0 = (__int64)&qword_4F822F0;
  v83 = (_QWORD *)qword_4F82348[9];
  dword_4F822E0 = 1065353216;
  v7 = &qword_4F822F0 - 2;
  qword_4F822E8 = 0;
  v8 = (__int64)(qword_4F82348[9] - qword_4F82348[8]) >> 5;
  qword_4F822F0 = 0;
  v9 = sub_222D860(&qword_4F822F0 - 2, v8);
  v12 = v9;
  if ( v9 > qword_4F822C8 )
  {
    if ( v9 == 1 )
    {
      qword_4F822F0 = 0;
      v14 = &qword_4F822F0;
    }
    else
    {
      if ( v9 > 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_99;
      v13 = (void *)sub_22077B0(8 * v9);
      v14 = (__int64 *)memset(v13, 0, 8 * v12);
    }
    qword_4F822C0 = (__int64)v14;
    qword_4F822C8 = v12;
  }
  for ( i = v6; v83 != i; i += 4 )
  {
    v16 = sub_22076E0(*i, i[1], 3339675911LL);
    v17 = qword_4F822C8;
    v18 = v16;
    v19 = v16 % qword_4F822C8;
    v20 = *(_QWORD **)(qword_4F822C0 + 8 * (v16 % qword_4F822C8));
    v21 = 8 * (v16 % qword_4F822C8);
    if ( !v20 )
      goto LABEL_63;
    v22 = (_QWORD *)*v20;
    v23 = 8 * v19;
    v24 = *(_QWORD **)(qword_4F822C0 + 8 * v19);
    v25 = i;
    v26 = v16 % qword_4F822C8;
    v27 = v22[5];
    while ( 1 )
    {
      if ( v18 == v27 )
      {
        v28 = v25[1];
        if ( v28 == v22[2] )
        {
          if ( !v28 )
            break;
          v84 = v23;
          v85 = v25;
          v29 = memcmp((const void *)*v25, (const void *)v22[1], v28);
          v25 = v85;
          v23 = v84;
          if ( !v29 )
            break;
        }
      }
      if ( !*v22 || (v27 = *(_QWORD *)(*v22 + 40LL), v24 = v22, v26 != v27 % v17) )
      {
        v21 = v23;
        i = v25;
LABEL_63:
        v64 = (__int64 *)sub_22077B0(48);
        v65 = v64;
        if ( v64 )
          *v64 = 0;
        v64[1] = (__int64)(v64 + 3);
        sub_BB8750(v64 + 1, (_BYTE *)*i, *i + i[1]);
        v8 = qword_4F822C8;
        v7 = (__int64 *)&dword_4F822E0;
        v66 = sub_222DA10(&dword_4F822E0, qword_4F822C8, qword_4F822D8, 1);
        v67 = v10;
        if ( !v66 )
        {
          v65[5] = v18;
          v68 = (__int64 *)qword_4F822C0;
          v69 = (_QWORD *)(qword_4F822C0 + v21);
          v70 = *(__int64 **)(qword_4F822C0 + v21);
          if ( v70 )
            goto LABEL_67;
LABEL_82:
          v78 = qword_4F822D0;
          qword_4F822D0 = (__int64)v65;
          *v65 = v78;
          if ( v78 )
          {
            v68[*(_QWORD *)(v78 + 40) % (unsigned __int64)qword_4F822C8] = (__int64)v65;
            v69 = (_QWORD *)(v21 + qword_4F822C0);
          }
          *v69 = &qword_4F822D0;
          goto LABEL_68;
        }
        if ( v10 != 1 )
        {
          if ( v10 <= 0xFFFFFFFFFFFFFFFLL )
          {
            v71 = 8 * v10;
            v72 = (void *)sub_22077B0(8 * v10);
            v68 = (__int64 *)memset(v72, 0, v71);
            goto LABEL_72;
          }
LABEL_99:
          sub_4261EA(v7, v8, v10, v11);
        }
        qword_4F822F0 = 0;
        v68 = &qword_4F822F0;
LABEL_72:
        v73 = (_QWORD *)qword_4F822D0;
        qword_4F822D0 = 0;
        if ( !v73 )
        {
LABEL_79:
          if ( (__int64 *)qword_4F822C0 != &qword_4F822F0 )
          {
            v86 = v68;
            j_j___libc_free_0(qword_4F822C0, 8 * qword_4F822C8);
            v68 = v86;
          }
          qword_4F822C8 = v67;
          qword_4F822C0 = (__int64)v68;
          v65[5] = v18;
          v21 = 8 * (v18 % v67);
          v69 = (__int64 *)((char *)v68 + v21);
          v70 = *(__int64 **)((char *)v68 + v21);
          if ( !v70 )
            goto LABEL_82;
LABEL_67:
          *v65 = *v70;
          *(_QWORD *)*v69 = v65;
LABEL_68:
          ++qword_4F822D8;
          goto LABEL_22;
        }
        v74 = 0;
        while ( 1 )
        {
          v75 = v73;
          v73 = (_QWORD *)*v73;
          v76 = v75[5] % v67;
          v77 = &v68[v76];
          if ( *v77 )
            break;
          *v75 = qword_4F822D0;
          qword_4F822D0 = (__int64)v75;
          *v77 = (__int64)&qword_4F822D0;
          if ( !*v75 )
          {
            v74 = v76;
LABEL_75:
            if ( !v73 )
              goto LABEL_79;
            continue;
          }
          v68[v74] = (__int64)v75;
          v74 = v76;
          if ( !v73 )
            goto LABEL_79;
        }
        *v75 = *(_QWORD *)*v77;
        *(_QWORD *)*v77 = v75;
        goto LABEL_75;
      }
      v22 = (_QWORD *)*v22;
    }
    i = v25;
    v30 = v24;
    v21 = v23;
    if ( !*v30 )
      goto LABEL_63;
LABEL_22:
    ;
  }
  v3 = 0;
  __cxa_atexit((void (*)(void *))sub_8565C0, &qword_4F822C0, &qword_4A427C0);
  sub_2207640(&byte_4F822A0);
  if ( qword_4F822D8 )
  {
LABEL_24:
    v31 = (_BYTE *)sub_BD5D20(a2);
    v94 = v96;
    sub_BB88C0((__int64 *)&v94, v31, (__int64)&v31[v32]);
    v3 = sub_BB97F0(&qword_4F822C0, (const void **)&v94) == 0;
    if ( v94 != v96 )
      j_j___libc_free_0(v94, *(_QWORD *)&v96[0] + 1LL);
  }
LABEL_3:
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v89 + 24LL))(v89);
  if ( !(_BYTE)v4 )
    return sub_B2D610(a2, 48);
  v33 = *(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, __m128i *, __int64))(*(_QWORD *)v89 + 16LL);
  v34 = (_BYTE *)sub_BD5D20(a2);
  if ( v34 )
  {
    v92[0] = (__int64)v93;
    sub_BB88C0(v92, v34, (__int64)&v34[v35]);
  }
  else
  {
    v92[1] = 0;
    v92[0] = (__int64)v93;
    LOBYTE(v93[0]) = 0;
  }
  v36 = sub_2241130(v92, 0, 0, "function (", 10);
  v94 = v96;
  if ( *(_QWORD *)v36 == v36 + 16 )
  {
    v96[0] = _mm_loadu_si128((const __m128i *)(v36 + 16));
  }
  else
  {
    v94 = *(_OWORD **)v36;
    *(_QWORD *)&v96[0] = *(_QWORD *)(v36 + 16);
  }
  v95 = *(_QWORD *)(v36 + 8);
  *(_QWORD *)v36 = v36 + 16;
  *(_QWORD *)(v36 + 8) = 0;
  *(_BYTE *)(v36 + 16) = 0;
  if ( v95 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  v37 = (__m128i *)sub_2241490(&v94, ")", 1, v96);
  v90 = &v91;
  if ( (__m128i *)v37->m128i_i64[0] == &v37[1] )
  {
    v91 = _mm_loadu_si128(v37 + 1);
  }
  else
  {
    v90 = (__m128i *)v37->m128i_i64[0];
    v91.m128i_i64[0] = v37[1].m128i_i64[0];
  }
  v38 = v37->m128i_i64[1];
  v37[1].m128i_i8[0] = 0;
  v37->m128i_i64[0] = (__int64)v37[1].m128i_i64;
  v37->m128i_i64[1] = 0;
  if ( v94 != v96 )
    j_j___libc_free_0(v94, *(_QWORD *)&v96[0] + 1LL);
  if ( (_QWORD *)v92[0] != v93 )
    j_j___libc_free_0(v92[0], v93[0] + 1LL);
  v39 = *(const char *(__fastcall **)(__int64, __int64))(*a1 + 16LL);
  if ( v39 == sub_BB8680 )
  {
    v40 = (pthread_rwlock_t *)sub_BC2B00(a1, a1[2]);
    v41 = sub_BC2C30(v40);
    v42 = v90;
    v43 = v38;
    v44 = 43;
    v45 = "Unnamed pass: implement Pass::getPassName()";
    if ( v41 )
    {
      v45 = *(const char **)v41;
      v44 = *(_QWORD *)(v41 + 8);
    }
  }
  else
  {
    v81 = ((__int64 (__fastcall *)(_QWORD *))v39)(a1);
    v43 = v38;
    v42 = v90;
    v45 = (const char *)v81;
  }
  v46 = (__int64)v90;
  if ( v33(v89, v45, v44, v42, v43) )
  {
    if ( v90 != &v91 )
      j_j___libc_free_0(v90, v91.m128i_i64[0] + 1);
    return sub_B2D610(a2, 48);
  }
  if ( v90 != &v91 )
  {
    v45 = (const char *)(v91.m128i_i64[0] + 1);
    j_j___libc_free_0(v90, v91.m128i_i64[0] + 1);
  }
  if ( v3 )
  {
    v47 = sub_CB72A0(v90, v45);
    v48 = *(__m128i **)(v47 + 32);
    v49 = v47;
    if ( *(_QWORD *)(v47 + 24) - (_QWORD)v48 <= 0x17u )
    {
      v46 = v47;
      v45 = "BISECT: Skip bisecting '";
      v49 = sub_CB6200(v47, "BISECT: Skip bisecting '", 24);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F560C0);
      v48[1].m128i_i64[0] = 0x2720676E69746365LL;
      *v48 = si128;
      *(_QWORD *)(v47 + 32) += 24LL;
    }
    v51 = *(const char *(__fastcall **)(__int64, __int64))(*a1 + 16LL);
    if ( v51 == sub_BB8680 )
    {
      v52 = (pthread_rwlock_t *)sub_BC2B00(v46, v45);
      v53 = sub_BC2C30(v52);
      if ( !v53 )
      {
        v56 = *(void **)(v49 + 32);
        v54 = "Unnamed pass: implement Pass::getPassName()";
        v55 = 43;
        if ( *(_QWORD *)(v49 + 24) - (_QWORD)v56 > 0x2Au )
          goto LABEL_94;
        goto LABEL_89;
      }
      v54 = *(const char **)v53;
      v55 = *(_QWORD *)(v53 + 8);
    }
    else
    {
      v54 = (const char *)((__int64 (__fastcall *)(_QWORD *))v51)(a1);
      v55 = v82;
    }
    v56 = *(void **)(v49 + 32);
    v57 = *(_QWORD *)(v49 + 24) - (_QWORD)v56;
    if ( v57 >= v55 )
    {
      if ( !v55 )
      {
LABEL_51:
        if ( v57 <= 0xD )
        {
          v49 = sub_CB6200(v49, "' on function ", 14);
        }
        else
        {
          qmemcpy(v56, "' on function ", 14);
          *(_QWORD *)(v49 + 32) += 14LL;
        }
        v58 = sub_BD5D20(a2);
        v60 = *(_BYTE **)(v49 + 32);
        v61 = (const void *)v58;
        v62 = *(_BYTE **)(v49 + 24);
        v63 = v59;
        if ( v59 > v62 - v60 )
        {
          v49 = sub_CB6200(v49, v61, v59);
          v62 = *(_BYTE **)(v49 + 24);
          v60 = *(_BYTE **)(v49 + 32);
        }
        else if ( v59 )
        {
          memcpy(v60, v61, v59);
          v62 = *(_BYTE **)(v49 + 24);
          v60 = (_BYTE *)(v63 + *(_QWORD *)(v49 + 32));
          *(_QWORD *)(v49 + 32) = v60;
        }
        if ( v62 == v60 )
        {
          sub_CB6200(v49, "\n", 1);
        }
        else
        {
          *v60 = 10;
          ++*(_QWORD *)(v49 + 32);
        }
        return sub_B2D610(a2, 48);
      }
LABEL_94:
      memcpy(v56, v54, v55);
      v80 = *(_QWORD *)(v49 + 24);
      v56 = (void *)(v55 + *(_QWORD *)(v49 + 32));
      *(_QWORD *)(v49 + 32) = v56;
      v57 = v80 - (_QWORD)v56;
      goto LABEL_51;
    }
LABEL_89:
    v79 = sub_CB6200(v49, v54, v55);
    v56 = *(void **)(v79 + 32);
    v49 = v79;
    v57 = *(_QWORD *)(v79 + 24) - (_QWORD)v56;
    goto LABEL_51;
  }
  return v4;
}
