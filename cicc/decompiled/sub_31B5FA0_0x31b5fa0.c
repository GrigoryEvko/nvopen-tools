// Function: sub_31B5FA0
// Address: 0x31b5fa0
//
void __fastcall sub_31B5FA0(__int64 a1, _BYTE *a2, size_t a3, __int64 a4)
{
  const void *v4; // r8
  unsigned __int64 v8; // rax
  unsigned __int64 *v9; // rdx
  __m128i *v10; // rax
  void (__fastcall *v11)(size_t *, __int64, __int64); // rax
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rax
  size_t v15; // r13
  char *v16; // rbx
  unsigned __int64 v17; // r10
  unsigned __int64 v18; // rdx
  int v19; // ecx
  unsigned __int64 v20; // r12
  int v21; // r14d
  char *v22; // r15
  char v23; // cl
  unsigned __int64 *v24; // rdi
  unsigned __int64 v25; // rcx
  __int64 v26; // r8
  unsigned __int64 v27; // r8
  char v28; // cl
  char v29; // si
  size_t v30; // r11
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // r13
  void *v34; // rax
  __int64 v35; // rax
  _QWORD *v36; // r8
  char *v37; // rax
  _QWORD *v38; // r8
  char *v39; // rax
  void *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  int v43; // [rsp-A4h] [rbp-A4h]
  unsigned __int64 v44; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v45; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v46; // [rsp-98h] [rbp-98h]
  unsigned __int64 v47; // [rsp-98h] [rbp-98h]
  unsigned __int64 v48; // [rsp-90h] [rbp-90h]
  const void *v49; // [rsp-90h] [rbp-90h]
  __m128i *v50; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v51; // [rsp-80h] [rbp-80h]
  __m128i v52; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 *v53; // [rsp-68h] [rbp-68h] BYREF
  size_t v54; // [rsp-60h] [rbp-60h] BYREF
  unsigned __int64 v55; // [rsp-58h] [rbp-58h] BYREF
  void (__fastcall *v56)(size_t *, size_t *, __int64); // [rsp-50h] [rbp-50h]
  __int64 v57; // [rsp-48h] [rbp-48h]

  if ( !a3 )
    return;
  v4 = a2;
  v53 = &v55;
  if ( !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v50 = (__m128i *)a3;
  if ( a3 > 0xF )
  {
    v35 = sub_22409D0((__int64)&v53, (unsigned __int64 *)&v50, 0);
    v4 = a2;
    v53 = (unsigned __int64 *)v35;
    v24 = (unsigned __int64 *)v35;
    v55 = (unsigned __int64)v50;
    goto LABEL_27;
  }
  v8 = a3;
  if ( a3 != 1 )
  {
    v24 = &v55;
LABEL_27:
    memcpy(v24, v4, a3);
    v8 = (unsigned __int64)v50;
    v9 = v53;
    goto LABEL_6;
  }
  LOBYTE(v55) = *a2;
  v9 = &v55;
LABEL_6:
  v54 = v8;
  *((_BYTE *)v9 + v8) = 0;
  v10 = (__m128i *)sub_2240FD0((unsigned __int64 *)&v53, v54, 0, 1u, 0);
  v50 = &v52;
  if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
  {
    v52 = _mm_loadu_si128(v10 + 1);
  }
  else
  {
    v50 = (__m128i *)v10->m128i_i64[0];
    v52.m128i_i64[0] = v10[1].m128i_i64[0];
  }
  v51 = v10->m128i_u64[1];
  v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
  v10->m128i_i64[1] = 0;
  v10[1].m128i_i8[0] = 0;
  if ( v53 != &v55 )
    j_j___libc_free_0((unsigned __int64)v53);
  v11 = *(void (__fastcall **)(size_t *, __int64, __int64))(a4 + 16);
  v12 = (unsigned __int64)v50;
  v53 = (unsigned __int64 *)a1;
  v56 = 0;
  v13 = v51;
  if ( v11 )
  {
    v48 = v51;
    v11(&v54, a4, 2);
    v13 = v48;
    v57 = *(_QWORD *)(a4 + 24);
    v56 = *(void (__fastcall **)(size_t *, size_t *, __int64))(a4 + 16);
  }
  v14 = v12 + v13;
  v15 = 0;
  v49 = 0;
  if ( v12 + v13 != v12 )
  {
    v16 = (char *)v12;
    v17 = v12;
    LODWORD(v18) = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = (char *)v14;
    do
    {
LABEL_18:
      if ( v19 != 1 )
        goto LABEL_38;
      while ( 1 )
      {
        v23 = *v16;
        if ( *v16 == 60 )
          break;
        if ( v23 != 62 )
        {
          if ( !v23 )
          {
            v40 = sub_CB72A0();
            v41 = sub_904010(
                    (__int64)v40,
                    "Missing '>' in pass pipeline. End-of-string reached while reading arguments for pass '");
            v42 = sub_A51340(v41, v49, v15);
            sub_904010(v42, "'.\n");
            exit(1);
          }
LABEL_16:
          v19 = 1;
LABEL_17:
          ++v16;
          ++v20;
          if ( v22 == v16 )
            goto LABEL_21;
          goto LABEL_18;
        }
        if ( --v21 )
        {
          if ( v21 < 0 )
          {
            v34 = sub_CB72A0();
            sub_904010((__int64)v34, "Unexpected '>' in pass pipeline.\n");
            exit(1);
          }
          goto LABEL_16;
        }
        v25 = v43;
        if ( v43 > v13 )
          v25 = v13;
        v26 = 0;
        if ( v20 >= v25 )
        {
          v27 = v13;
          if ( v20 <= v13 )
            v27 = v20;
          v26 = v27 - v25;
        }
        v44 = v13;
        v46 = v17;
        sub_31B5DF0((__int64 *)&v53, v49, v15, v17 + v25, v26);
        v17 = v46;
        v13 = v44;
        if ( v22 == v16 + 1 )
          goto LABEL_21;
        v28 = v16[1];
        if ( v28 && v28 != 44 )
        {
          v38 = sub_CB72A0();
          v39 = (char *)v38[4];
          if ( v38[3] - (_QWORD)v39 <= 0x39u )
          {
            sub_CB6200((__int64)v38, "Expected delimiter or end-of-string after pass arguments.\n", 0x3Au);
          }
          else
          {
            qmemcpy(v39, "Expected delimiter or end-of-string after pass arguments.\n", 0x3Au);
            v38[4] += 58LL;
          }
          exit(1);
        }
        v16 += 2;
        LODWORD(v18) = v20 + 2;
        v20 += 2LL;
        if ( v22 == v16 )
          goto LABEL_21;
LABEL_38:
        v29 = *v16;
        if ( *v16 != 60 )
        {
          if ( v29 == 62 )
          {
            v36 = sub_CB72A0();
            v37 = (char *)v36[4];
            if ( v36[3] - (_QWORD)v37 <= 0x20u )
            {
              sub_CB6200((__int64)v36, "Unexpected '>' in pass pipeline.\n", 0x21u);
            }
            else
            {
              qmemcpy(v37, "Unexpected '>' in pass pipeline.\n", 0x21u);
              v36[4] += 33LL;
            }
            exit(1);
          }
          if ( !v29 || (v19 = 0, v29 == 44) )
          {
            v18 = (int)v18;
            if ( (int)v18 > v13 )
              v18 = v13;
            v30 = 0;
            if ( v20 >= v18 )
            {
              v31 = v20;
              if ( v13 <= v20 )
                v31 = v13;
              v30 = v31 - v18;
            }
            v45 = v13;
            v47 = v17;
            sub_31B5DF0((__int64 *)&v53, (const void *)(v17 + v18), v30, 0, 0);
            v17 = v47;
            v13 = v45;
            v19 = 0;
            LODWORD(v18) = v20 + 1;
          }
          goto LABEL_17;
        }
        v32 = (int)v18;
        if ( (int)v18 > v13 )
          v32 = v13;
        v33 = v32;
        if ( v20 >= v32 )
        {
          v33 = v20;
          if ( v13 <= v20 )
            v33 = v13;
        }
        ++v16;
        v15 = v33 - v32;
        v49 = (const void *)(v17 + v32);
        ++v21;
        v43 = ++v20;
        if ( v22 == v16 )
          goto LABEL_21;
      }
      ++v16;
      ++v21;
      v19 = 1;
      ++v20;
    }
    while ( v22 != v16 );
  }
LABEL_21:
  if ( v56 )
    v56(&v54, &v54, 3);
  if ( v50 != &v52 )
    j_j___libc_free_0((unsigned __int64)v50);
}
