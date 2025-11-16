// Function: sub_3976C50
// Address: 0x3976c50
//
void __fastcall sub_3976C50(__int64 a1)
{
  __int64 v2; // rax
  __m128i *v3; // rdx
  __int64 v4; // r12
  __m128i si128; // xmm0
  const char *v6; // rax
  size_t v7; // rdx
  char *v8; // rsi
  size_t v9; // r13
  unsigned __int64 v10; // rax
  _BYTE *v11; // rdi
  __int64 v12; // r13
  void (__fastcall *v13)(__int64, __int64, _QWORD); // r15
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // r12
  __int64 v23; // rsi
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // rax
  __int64 *v27; // r12
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // rsi
  __int64 v34; // rdi
  void (*v35)(); // rcx
  __int64 v36; // rsi
  _QWORD *v37; // rax
  _QWORD *v38; // r12
  const char *v39; // rdi
  unsigned int v40; // r13d
  char *v41; // r14
  size_t v42; // rax
  char *v43; // rdi
  double *v44; // r15
  size_t v45; // rax
  const char *v46; // rcx
  __int64 v47; // r8
  size_t v48; // rax
  unsigned __int8 *v49; // rsi
  size_t v50; // rdx
  size_t v51; // rax
  __int64 *v52; // r12
  __int64 v53; // rsi
  __int64 v54; // r12
  __int64 v55; // r13
  void (__fastcall *v56)(__int64, _QWORD, __int64); // r14
  __int64 v57; // rax
  __int64 *v58; // r12
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // [rsp+0h] [rbp-B0h]
  _QWORD *v62; // [rsp+10h] [rbp-A0h]
  int v63[2]; // [rsp+18h] [rbp-98h]
  int v64[2]; // [rsp+20h] [rbp-90h]
  int v65[2]; // [rsp+28h] [rbp-88h]
  size_t n; // [rsp+38h] [rbp-78h]
  unsigned __int64 v67; // [rsp+40h] [rbp-70h] BYREF
  __int64 v68; // [rsp+48h] [rbp-68h]
  __int64 v69; // [rsp+50h] [rbp-60h]
  __m128i *v70[2]; // [rsp+60h] [rbp-50h] BYREF
  char v71; // [rsp+70h] [rbp-40h]
  char v72; // [rsp+71h] [rbp-3Fh]

  v61 = **(_QWORD **)(a1 + 264);
  if ( *(_BYTE *)(a1 + 416) )
  {
    v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
    v3 = *(__m128i **)(v2 + 24);
    v4 = v2;
    if ( *(_QWORD *)(v2 + 16) - (_QWORD)v3 <= 0x11u )
    {
      v4 = sub_16E7EE0(v2, "-- Begin function ", 0x12u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44D4130);
      v3[1].m128i_i16[0] = 8302;
      *v3 = si128;
      *(_QWORD *)(v2 + 24) += 18LL;
    }
    v6 = sub_1649960(v61);
    v8 = (char *)v6;
    v9 = v7;
    if ( !v7 )
    {
      v10 = *(_QWORD *)(v4 + 16);
      v11 = *(_BYTE **)(v4 + 24);
      goto LABEL_6;
    }
    if ( *v6 == 1 )
    {
      v10 = *(_QWORD *)(v4 + 16);
      v11 = *(_BYTE **)(v4 + 24);
      v9 = v7 - 1;
      ++v8;
      if ( v10 - (unsigned __int64)v11 < v7 - 1 )
      {
LABEL_50:
        v60 = sub_16E7EE0(v4, v8, v9);
        v11 = *(_BYTE **)(v60 + 24);
        v4 = v60;
        if ( *(_QWORD *)(v60 + 16) > (unsigned __int64)v11 )
          goto LABEL_7;
LABEL_51:
        sub_16E7DE0(v4, 10);
        goto LABEL_8;
      }
      if ( v7 == 1 )
      {
LABEL_6:
        if ( v10 > (unsigned __int64)v11 )
        {
LABEL_7:
          *(_QWORD *)(v4 + 24) = v11 + 1;
          *v11 = 10;
          goto LABEL_8;
        }
        goto LABEL_51;
      }
    }
    else
    {
      v11 = *(_BYTE **)(v4 + 24);
      if ( *(_QWORD *)(v4 + 16) - (_QWORD)v11 < v7 )
        goto LABEL_50;
    }
    memcpy(v11, v8, v9);
    v10 = *(_QWORD *)(v4 + 16);
    v11 = (_BYTE *)(v9 + *(_QWORD *)(v4 + 24));
    *(_QWORD *)(v4 + 24) = v11;
    goto LABEL_6;
  }
LABEL_8:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 192LL))(a1);
  v12 = *(_QWORD *)(a1 + 256);
  v13 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v12 + 160LL);
  v14 = sub_396DD80(a1);
  v15 = *(_QWORD *)(a1 + 232);
  v16 = v14;
  LOBYTE(v19) = sub_394AFB0(v61, v15, v17, v18);
  v20 = sub_394B210(v16, v61, v19, v15);
  v13(v12, v20, 0);
  sub_39719F0(a1, *(_QWORD *)(a1 + 304), (*(_BYTE *)(v61 + 32) >> 4) & 3, 1);
  sub_396E9D0(a1, (_BYTE *)v61, *(_QWORD *)(a1 + 304));
  v21 = *(_QWORD *)(a1 + 240);
  if ( *(_BYTE *)(v21 + 304) )
  {
    sub_396F480(a1, *(_DWORD *)(*(_QWORD *)(a1 + 264) + 340LL), v61);
    v21 = *(_QWORD *)(a1 + 240);
  }
  if ( *(_BYTE *)(v21 + 305) )
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
      *(_QWORD *)(a1 + 256),
      *(_QWORD *)(a1 + 304),
      1);
  if ( *(_BYTE *)(a1 + 416) )
  {
    v22 = *(_QWORD **)(v61 + 40);
    v23 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
    sub_15537D0(v61, v23, 0, v22);
    v24 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 256) + 112LL))(*(_QWORD *)(a1 + 256));
    v25 = *(_BYTE **)(v24 + 24);
    if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 16) )
    {
      sub_16E7DE0(v24, 10);
    }
    else
    {
      *(_QWORD *)(v24 + 24) = v25 + 1;
      *v25 = 10;
    }
  }
  if ( (*(_BYTE *)(v61 + 18) & 2) != 0 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 18LL) )
    {
      v26 = sub_38BFE40(*(_QWORD *)(a1 + 248));
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(*(_QWORD *)(a1 + 256), v26, 0);
      v27 = (__int64 *)sub_15E3920(v61);
      v28 = sub_1632FA0(*(_QWORD *)(v61 + 40));
      sub_3976960(a1, v28, v27);
      (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 256) + 256LL))(
        *(_QWORD *)(a1 + 256),
        *(_QWORD *)(a1 + 304),
        16);
    }
    else
    {
      v52 = (__int64 *)sub_15E3920(v61);
      v53 = sub_1632FA0(*(_QWORD *)(v61 + 40));
      sub_3976960(a1, v53, v52);
    }
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 296LL))(a1);
  v29 = *(_QWORD *)(a1 + 272);
  v67 = 0;
  v68 = 0;
  v69 = 0;
  sub_1E2CED0(v29, v61, (__int64 *)&v67);
  v30 = (__int64)(v68 - v67) >> 3;
  if ( (_DWORD)v30 )
  {
    v31 = 0;
    v32 = 8LL * (unsigned int)v30;
    do
    {
      v34 = *(_QWORD *)(a1 + 256);
      v35 = *(void (**)())(*(_QWORD *)v34 + 104LL);
      v72 = 1;
      v70[0] = (__m128i *)"Address taken block that was later removed";
      v71 = 3;
      if ( v35 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64, __m128i **, __int64))v35)(v34, v70, 1);
        v34 = *(_QWORD *)(a1 + 256);
      }
      v33 = *(_QWORD *)(v67 + v31);
      v31 += 8;
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v34 + 176LL))(v34, v33, 0);
    }
    while ( v32 != v31 );
  }
  v36 = *(_QWORD *)(a1 + 384);
  if ( v36 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 72LL) )
    {
      v54 = sub_38BFA60(*(_QWORD *)(a1 + 248), 1);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(*(_QWORD *)(a1 + 256), v54, 0);
      v55 = *(_QWORD *)(a1 + 256);
      v56 = *(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v55 + 240LL);
      v57 = sub_38CF310(v54, 0, *(_QWORD *)(a1 + 248), 0);
      v56(v55, *(_QWORD *)(a1 + 384), v57);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(*(_QWORD *)(a1 + 256), v36, 0);
    }
  }
  v37 = *(_QWORD **)(a1 + 424);
  v62 = &v37[5 * *(unsigned int *)(a1 + 432)];
  if ( v37 != v62 )
  {
    v38 = *(_QWORD **)(a1 + 424);
    do
    {
      v39 = (const char *)v38[4];
      v40 = unk_4F9E388;
      v41 = (char *)v39;
      v42 = 0;
      if ( v39 )
        v42 = strlen(v39);
      v43 = (char *)v38[3];
      v44 = (double *)v42;
      v45 = 0;
      if ( v43 )
        v45 = strlen(v43);
      v46 = (const char *)v38[2];
      n = v45;
      v47 = 0;
      if ( v46 )
      {
        *(_QWORD *)v65 = v38[2];
        v48 = strlen(v46);
        v46 = *(const char **)v65;
        v47 = v48;
      }
      v49 = (unsigned __int8 *)v38[1];
      v50 = 0;
      if ( v49 )
      {
        *(_QWORD *)v63 = v46;
        *(_QWORD *)v64 = v47;
        v51 = strlen((const char *)v38[1]);
        v46 = *(const char **)v63;
        v47 = *(_QWORD *)v64;
        v50 = v51;
      }
      sub_16D8B50(v70, v49, v50, (__int64)v46, v47, v40, (unsigned __int8 *)v43, n, v41, v44);
      (*(void (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)*v38 + 32LL))(*v38, *(_QWORD *)(a1 + 264));
      if ( v70[0] )
        sub_16D7950((__int64)v70[0]->m128i_i64);
      v38 += 5;
    }
    while ( v62 != v38 );
  }
  if ( (*(_BYTE *)(v61 + 18) & 4) != 0 )
  {
    v58 = (__int64 *)sub_15E3950(v61);
    v59 = sub_1632FA0(*(_QWORD *)(v61 + 40));
    sub_3976960(a1, v59, v58);
  }
  if ( v67 )
    j_j___libc_free_0(v67);
}
