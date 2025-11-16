// Function: sub_31EB340
// Address: 0x31eb340
//
void __fastcall sub_31EB340(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // r13
  __m128i si128; // xmm0
  const char *v7; // rax
  size_t v8; // rdx
  _BYTE *v9; // rdi
  size_t v10; // r14
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  unsigned __int8 v32; // dl
  __int64 v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rsi
  void (*v40)(); // rax
  __int64 v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 *v46; // rbx
  __int64 *i; // r13
  __int64 v48; // rdi
  __int64 v49; // r9
  void (*v50)(); // rcx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 *v54; // rbx
  __int64 *v55; // r13
  __int64 v56; // rdi
  __int64 *v57; // rbx
  __int64 *v58; // r13
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // r14
  void (__fastcall *v63)(__int64, _QWORD, unsigned __int64); // rbx
  unsigned __int64 v64; // rax
  __int64 v65; // r13
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // [rsp+0h] [rbp-90h]
  unsigned __int64 v70; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+20h] [rbp-70h]
  unsigned __int64 v73[4]; // [rsp+30h] [rbp-60h] BYREF
  char v74; // [rsp+50h] [rbp-40h]
  char v75; // [rsp+51h] [rbp-3Fh]

  v2 = **(_QWORD **)(a1 + 232);
  if ( !*(_BYTE *)(a1 + 488) )
    goto LABEL_10;
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 128LL))(*(_QWORD *)(a1 + 224));
  v4 = *(__m128i **)(v3 + 32);
  v5 = v3;
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x11u )
  {
    v5 = sub_CB6200(v3, "-- Begin function ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44D4130);
    v4[1].m128i_i16[0] = 8302;
    *v4 = si128;
    *(_QWORD *)(v3 + 32) += 18LL;
  }
  v7 = sub_BD5D20(v2);
  v9 = *(_BYTE **)(v5 + 32);
  v10 = v8;
  v11 = (unsigned __int8 *)v7;
  if ( v8 )
  {
    v12 = *(_QWORD *)(v5 + 24) - (_QWORD)v9;
    if ( *v11 != 1 )
    {
      if ( v12 < v8 )
      {
LABEL_7:
        v13 = sub_CB6200(v5, v11, v10);
        v9 = *(_BYTE **)(v13 + 32);
        v5 = v13;
        goto LABEL_8;
      }
      goto LABEL_71;
    }
    v10 = v8 - 1;
    ++v11;
    if ( v12 < v8 - 1 )
      goto LABEL_7;
    if ( v8 != 1 )
    {
LABEL_71:
      memcpy(v9, v11, v10);
      v9 = (_BYTE *)(v10 + *(_QWORD *)(v5 + 32));
      *(_QWORD *)(v5 + 32) = v9;
      if ( *(_QWORD *)(v5 + 24) > (unsigned __int64)v9 )
        goto LABEL_9;
      goto LABEL_72;
    }
  }
LABEL_8:
  if ( *(_QWORD *)(v5 + 24) > (unsigned __int64)v9 )
  {
LABEL_9:
    *(_QWORD *)(v5 + 32) = v9 + 1;
    *v9 = 10;
    goto LABEL_10;
  }
LABEL_72:
  sub_CB5D20(v5, 10);
LABEL_10:
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 208LL))(a1);
  v14 = *(_QWORD *)(a1 + 232);
  if ( *(_BYTE *)(*(_QWORD *)(v14 + 328) + 260LL) )
  {
    v15 = sub_31DA6B0(a1);
    *(_QWORD *)(v14 + 72) = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v15 + 80LL))(
                              v15,
                              v2,
                              *(_QWORD *)(a1 + 200));
  }
  else
  {
    v60 = sub_31DA6B0(a1);
    *(_QWORD *)(v14 + 72) = sub_3157D60(v60, v2, *(_QWORD *)(a1 + 200));
  }
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
    *(_QWORD *)(a1 + 224),
    *(_QWORD *)(*(_QWORD *)(a1 + 232) + 72LL),
    0);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 21LL) )
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 560LL))(a1, v2, *(_QWORD *)(a1 + 288));
  else
    sub_31DE970(a1, *(_QWORD *)(a1 + 280), (*(_BYTE *)(v2 + 32) >> 4) & 3, 1);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 560LL))(a1, v2, *(_QWORD *)(a1 + 280));
  v16 = *(_QWORD *)(a1 + 208);
  if ( *(_BYTE *)(v16 + 288) )
  {
    sub_31DCA70(a1, *(unsigned __int8 *)(*(_QWORD *)(a1 + 232) + 340LL), v2, 0);
    v16 = *(_QWORD *)(a1 + 208);
  }
  if ( *(_BYTE *)(v16 + 289) )
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(a1 + 280),
      2);
  if ( (unsigned __int8)sub_B2D610(v2, 5) )
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(a1 + 280),
      1);
  if ( (*(_BYTE *)(v2 + 2) & 2) != 0 )
  {
    v73[0] = sub_B2E510(v2);
    sub_31EB240(a1, (__int64)v73, 1, v67, v68);
  }
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 200LL))(a1, *(_QWORD *)(a1 + 232));
  v70 = sub_B2D7E0(v2, "patchable-function-prefix", 0x19u);
  v17 = sub_A72240((__int64 *)&v70);
  if ( sub_C93C90(v17, v18, 0xAu, v73) || (v19 = v73[0], v73[0] != LODWORD(v73[0])) )
  {
    v70 = sub_B2D7E0(v2, "patchable-function-entry", 0x18u);
    v51 = sub_A72240((__int64 *)&v70);
    if ( sub_C93C90(v51, v52, 0xAu, v73) )
      goto LABEL_28;
    v26 = v73[0];
    if ( v73[0] != LODWORD(v73[0]) )
      goto LABEL_28;
    goto LABEL_47;
  }
  v70 = sub_B2D7E0(v2, "patchable-function-entry", 0x18u);
  v20 = sub_A72240((__int64 *)&v70);
  v22 = v21;
  if ( !sub_C93C90(v20, v21, 0xAu, v73) )
  {
    v26 = v73[0];
    v23 = LODWORD(v73[0]);
    if ( v73[0] == LODWORD(v73[0]) )
    {
      if ( v19 )
        goto LABEL_27;
LABEL_47:
      if ( v26 )
        *(_QWORD *)(a1 + 272) = *(_QWORD *)(a1 + 536);
      goto LABEL_28;
    }
  }
  if ( v19 )
  {
LABEL_27:
    v27 = sub_E6C350(*(_QWORD *)(a1 + 216), v22, v23, v24, v25);
    v28 = *(_QWORD *)(a1 + 224);
    *(_QWORD *)(a1 + 272) = v27;
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v28 + 208LL))(v28, v27, 0);
    sub_31DCBB0(a1, v19);
  }
LABEL_28:
  if ( (*(_BYTE *)(v2 + 7) & 0x20) != 0 )
  {
    v29 = sub_B91C10(v2, 32);
    if ( v29 )
    {
      v32 = *(_BYTE *)(v29 - 16);
      if ( (v32 & 2) != 0 )
      {
        v35 = *(_QWORD **)(v29 - 32);
        v36 = *(_QWORD *)(*v35 + 136LL);
      }
      else
      {
        v33 = v29 - 8LL * ((v32 >> 2) & 0xF);
        v34 = *(_QWORD *)(v33 - 16);
        v35 = (_QWORD *)(v33 - 16);
        v36 = *(_QWORD *)(v34 + 136);
      }
      v37 = *(_QWORD *)(v35[1] + 136LL);
      v73[0] = v36;
      v73[1] = v37;
      sub_31EB240(a1, (__int64)v73, 2, v30, v31);
    }
  }
  if ( *(_BYTE *)(a1 + 488) )
  {
    v38 = *(_QWORD *)(v2 + 40);
    v39 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 128LL))(*(_QWORD *)(a1 + 224));
    sub_A5BF40((unsigned __int8 *)v2, v39, 0, v38);
    v40 = *(void (**)())(*(_QWORD *)a1 + 576LL);
    if ( v40 != nullsub_1846 )
      ((void (__fastcall *)(__int64))v40)(a1);
    v41 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 224) + 128LL))(*(_QWORD *)(a1 + 224));
    v42 = *(_BYTE **)(v41 + 32);
    if ( (unsigned __int64)v42 >= *(_QWORD *)(v41 + 24) )
    {
      sub_CB5D20(v41, 10);
    }
    else
    {
      *(_QWORD *)(v41 + 32) = v42 + 1;
      *v42 = 10;
    }
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 21LL) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 336LL))(a1);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 328LL))(a1);
  v70 = 0;
  v71 = 0;
  v72 = 0;
  sub_31DA230(a1, v2, &v70);
  v46 = v71;
  for ( i = (__int64 *)v70; v46 != i; ++i )
  {
    v48 = *(_QWORD *)(a1 + 224);
    v49 = *i;
    v50 = *(void (**)())(*(_QWORD *)v48 + 120LL);
    v75 = 1;
    v73[0] = (unsigned __int64)"Address taken block that was later removed";
    v74 = 3;
    if ( v50 != nullsub_98 )
    {
      v69 = v49;
      ((void (__fastcall *)(__int64, unsigned __int64 *, __int64))v50)(v48, v73, 1);
      v48 = *(_QWORD *)(a1 + 224);
      v49 = v69;
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v48 + 208LL))(v48, v49, 0);
  }
  v53 = *(_QWORD *)(a1 + 536);
  if ( v53 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 80LL) )
    {
      v61 = sub_E6C430(*(_QWORD *)(a1 + 216), v53, v43, v44, v45);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v61, 0);
      v62 = *(_QWORD *)(a1 + 224);
      v63 = *(void (__fastcall **)(__int64, _QWORD, unsigned __int64))(*(_QWORD *)v62 + 272LL);
      v64 = sub_E808D0(v61, 0, *(_QWORD **)(a1 + 216), 0);
      v63(v62, *(_QWORD *)(a1 + 536), v64);
    }
    else
    {
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v53, 0);
    }
  }
  v54 = *(__int64 **)(a1 + 576);
  v55 = &v54[*(unsigned int *)(a1 + 584)];
  while ( v55 != v54 )
  {
    v56 = *v54++;
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v56 + 32LL))(v56, *(_QWORD *)(a1 + 232));
    (*(void (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)*(v54 - 1) + 56LL))(
      *(v54 - 1),
      *(_QWORD *)(*(_QWORD *)(a1 + 232) + 328LL));
  }
  v57 = *(__int64 **)(a1 + 552);
  v58 = &v57[*(unsigned int *)(a1 + 560)];
  while ( v58 != v57 )
  {
    v59 = *v57++;
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v59 + 32LL))(v59, *(_QWORD *)(a1 + 232));
    (*(void (__fastcall **)(_QWORD, _QWORD))(*(_QWORD *)*(v57 - 1) + 56LL))(
      *(v57 - 1),
      *(_QWORD *)(*(_QWORD *)(a1 + 232) + 328LL));
  }
  if ( (*(_BYTE *)(v2 + 2) & 4) != 0 )
  {
    v65 = sub_B2E520(v2);
    v66 = sub_B2BEC0(v2);
    sub_31EA6F0(a1, v66, v65, 0);
  }
  if ( v70 )
    j_j___libc_free_0(v70);
}
