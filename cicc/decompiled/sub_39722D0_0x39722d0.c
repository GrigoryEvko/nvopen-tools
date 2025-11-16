// Function: sub_39722D0
// Address: 0x39722d0
//
__int64 __fastcall sub_39722D0(__int64 a1, __m128i *a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  void (*v7)(); // rax
  __int64 v8; // rax
  __m128i *v9; // rsi
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 v12; // r13
  __int64 *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdi
  void (*v16)(); // rax
  __int64 v17; // rdx
  int v18; // eax
  bool v19; // cc
  __int64 v20; // rax
  __int64 v21; // rbx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  __m128i *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rbx
  __int64 v31; // rax
  __m128i *v32; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  int v38; // r8d
  int v39; // r9d
  __int64 v40; // rax
  __m128i *v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r14
  __int64 (__fastcall *v45)(__m128i *, __m128i *, __int64, __int64, __int64); // r12
  const __m128i *v46; // r12
  __int64 v47; // rdi
  void (*v48)(); // rax
  void (*v49)(); // rax
  unsigned int v50; // eax
  _BYTE *v51; // rsi
  __int64 v52; // rdx
  __m128i *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rdi
  void (*v56)(); // rax
  void (*v57)(); // rax
  __int64 v58; // r12
  void (__fastcall *v59)(__int64, __int64); // r13
  __int64 v60; // rax
  __int64 *v61; // r12
  __int64 *v62; // rbx
  __int64 v63; // [rsp-10h] [rbp-D0h]
  __int64 v64; // [rsp+8h] [rbp-B8h]
  __int64 v65; // [rsp+10h] [rbp-B0h]
  unsigned int v66; // [rsp+10h] [rbp-B0h]
  __m128i *v67; // [rsp+18h] [rbp-A8h]
  __int64 v68; // [rsp+18h] [rbp-A8h]
  _QWORD v69[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v70; // [rsp+30h] [rbp-90h] BYREF
  __int16 v71; // [rsp+40h] [rbp-80h]
  __m128i v72; // [rsp+50h] [rbp-70h] BYREF
  __m128i v73; // [rsp+60h] [rbp-60h] BYREF
  char *v74; // [rsp+70h] [rbp-50h]

  v4 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC6A0E, 1u);
  v5 = v4;
  if ( v4 )
    v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_4FC6A0E);
  *(_QWORD *)(a1 + 272) = v5;
  v6 = sub_396DD80(a1);
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6 + 16LL))(
    v6,
    *(_QWORD *)(a1 + 248),
    *(_QWORD *)(a1 + 232));
  (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 168LL))(*(_QWORD *)(a1 + 256), 0);
  sub_38DDF50(*(_QWORD *)(a1 + 256), *(_QWORD *)(a1 + 232) + 472LL);
  v7 = *(void (**)())(*(_QWORD *)a1 + 224LL);
  if ( v7 != nullsub_779 )
    ((void (__fastcall *)(__int64, __m128i *))v7)(a1, a2);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 306LL) )
  {
    v58 = *(_QWORD *)(a1 + 256);
    v59 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v58 + 552LL);
    v60 = sub_16C40A0(a2[13].m128i_i64[0], a2[13].m128i_i64[1], 2);
    v59(v58, v60);
  }
  v8 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC3606, 1u);
  if ( !v8 )
    BUG();
  v9 = (__m128i *)&unk_4FC3606;
  v10 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4FC3606);
  v11 = *(__int64 **)(v10 + 160);
  v12 = v10;
  v13 = &v11[*(unsigned int *)(v10 + 168)];
  while ( v13 != v11 )
  {
    while ( 1 )
    {
      v9 = (__m128i *)*v11;
      v14 = sub_3971E20(a1, *v11);
      v15 = v14;
      if ( v14 )
      {
        v16 = *(void (**)())(*(_QWORD *)v14 + 16LL);
        if ( v16 != nullsub_1972 )
          break;
      }
      if ( v13 == ++v11 )
        goto LABEL_14;
    }
    ++v11;
    v9 = a2;
    ((void (__fastcall *)(__int64, __m128i *, __int64, __int64))v16)(v15, a2, v12, a1);
  }
LABEL_14:
  if ( a2[6].m128i_i64[0] )
  {
    v42 = *(_QWORD **)(a1 + 232);
    v43 = v42[60];
    v44 = v42[67];
    v64 = v42[70];
    v45 = *(__int64 (__fastcall **)(__m128i *, __m128i *, __int64, __int64, __int64))(v42[1] + 80LL);
    v65 = v42[71];
    v67 = (__m128i *)v42[66];
    v69[0] = v42[59];
    v69[1] = v43;
    if ( v45 )
    {
      v71 = 261;
      v70 = v69;
      sub_16E1010((__int64)&v72, (__int64)&v70);
      v9 = v67;
      v46 = (const __m128i *)v45(&v72, v67, v44, v64, v65);
      if ( (__m128i *)v72.m128i_i64[0] != &v73 )
      {
        v9 = (__m128i *)(v73.m128i_i64[0] + 1);
        j_j___libc_free_0(v72.m128i_u64[0]);
      }
    }
    else
    {
      v46 = 0;
    }
    v47 = *(_QWORD *)(a1 + 256);
    v48 = *(void (**)())(*(_QWORD *)v47 + 104LL);
    v72.m128i_i64[0] = (__int64)"Start of file scope inline assembly";
    v73.m128i_i16[0] = 259;
    if ( v48 != nullsub_580 )
    {
      v9 = &v72;
      ((void (__fastcall *)(__int64, __m128i *, __int64))v48)(v47, &v72, 1);
      v47 = *(_QWORD *)(a1 + 256);
    }
    v49 = *(void (**)())(*(_QWORD *)v47 + 144LL);
    if ( v49 != nullsub_581 )
      ((void (__fastcall *)(__int64, __m128i *))v49)(v47, v9);
    v68 = *(_QWORD *)(a1 + 232) + 840LL;
    v50 = (unsigned int)sub_38BE440(*(_QWORD *)(a1 + 248), v46);
    v51 = (_BYTE *)a2[5].m128i_i64[1];
    v66 = v50;
    v52 = (__int64)&v51[a2[6].m128i_i64[0]];
    v72.m128i_i64[0] = (__int64)&v73;
    sub_396BEA0(v72.m128i_i64, v51, v52);
    if ( v72.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v72, "\n", 1u);
    v53 = (__m128i *)v72.m128i_i64[0];
    sub_397D680(a1, v72.m128i_i32[0], v72.m128i_i32[2], v66, v68, 0, 0);
    v54 = v63;
    if ( (__m128i *)v72.m128i_i64[0] != &v73 )
    {
      v53 = (__m128i *)(v73.m128i_i64[0] + 1);
      j_j___libc_free_0(v72.m128i_u64[0]);
    }
    v55 = *(_QWORD *)(a1 + 256);
    v56 = *(void (**)())(*(_QWORD *)v55 + 104LL);
    v72.m128i_i64[0] = (__int64)"End of file scope inline assembly";
    v73.m128i_i16[0] = 259;
    if ( v56 != nullsub_580 )
    {
      v53 = &v72;
      ((void (__fastcall *)(__int64, __m128i *, __int64))v56)(v55, &v72, 1);
      v55 = *(_QWORD *)(a1 + 256);
    }
    v57 = *(void (**)())(*(_QWORD *)v55 + 144LL);
    if ( v57 != nullsub_581 )
      ((void (__fastcall *)(__int64, __m128i *, __int64))v57)(v55, v53, v54);
    if ( v46 )
      (*(void (__fastcall **)(const __m128i *, __m128i *, __int64))(v46->m128i_i64[0] + 8))(v46, v53, v54);
  }
  v17 = *(_QWORD *)(a1 + 240);
  if ( *(_BYTE *)(v17 + 344) )
  {
    v36 = sub_22077B0(0x1A50u);
    v37 = v36;
    if ( v36 )
      sub_3988FF0(v36, a1, a2);
    *(_QWORD *)(a1 + 504) = v37;
    sub_399B1E0(v37);
    v72.m128i_i64[0] = *(_QWORD *)(a1 + 504);
    v72.m128i_i64[1] = (__int64)"emit";
    v73.m128i_i64[0] = (__int64)"Debug Info Emission";
    v73.m128i_i64[1] = (__int64)"dwarf";
    v74 = "DWARF Emission";
    v40 = *(unsigned int *)(a1 + 432);
    if ( (unsigned int)v40 >= *(_DWORD *)(a1 + 436) )
    {
      sub_16CD150(a1 + 424, (const void *)(a1 + 440), 0, 40, v38, v39);
      v40 = *(unsigned int *)(a1 + 432);
    }
    v41 = (__m128i *)(*(_QWORD *)(a1 + 424) + 40 * v40);
    *v41 = _mm_loadu_si128(&v72);
    v41[1] = _mm_loadu_si128(&v73);
    v41[2].m128i_i64[0] = (__int64)v74;
    v17 = *(_QWORD *)(a1 + 240);
    ++*(_DWORD *)(a1 + 432);
  }
  if ( (unsigned int)(*(_DWORD *)(v17 + 348) - 1) <= 2 )
  {
    *(_BYTE *)(a1 + 536) = 1;
    v18 = *(_DWORD *)(v17 + 348);
    if ( v18 == 1 )
    {
      v61 = (__int64 *)a2[2].m128i_i64[0];
      v62 = &a2[1].m128i_i64[1];
      if ( v62 == v61 )
      {
LABEL_21:
        v20 = sub_22077B0(0x20u);
        v21 = v20;
        if ( !v20 )
          goto LABEL_26;
        sub_3983DA0(v20, a1);
        goto LABEL_23;
      }
      while ( 1 )
      {
        if ( !v61 )
          BUG();
        if ( (*(_BYTE *)(v61 - 3) & 0xF) != 1
          && !sub_15E4F60((__int64)(v61 - 7))
          && ((unsigned __int8)sub_1560180((__int64)(v61 + 7), 56)
           || !(unsigned __int8)sub_1560180((__int64)(v61 + 7), 30)
           || (*((_BYTE *)v61 - 38) & 8) != 0) )
        {
          break;
        }
        v61 = (__int64 *)v61[1];
        if ( v62 == v61 )
          goto LABEL_70;
      }
      *(_BYTE *)(a1 + 536) = 0;
LABEL_70:
      v17 = *(_QWORD *)(a1 + 240);
      v18 = *(_DWORD *)(v17 + 348);
    }
    v19 = v18 <= 3;
    if ( v18 != 3 )
      goto LABEL_19;
LABEL_35:
    v34 = sub_22077B0(0x20u);
    v21 = v34;
    if ( !v34 )
      goto LABEL_26;
    sub_39C08D0(v34, a1);
    goto LABEL_23;
  }
  *(_BYTE *)(a1 + 536) = 0;
  v18 = *(_DWORD *)(v17 + 348);
  v19 = v18 <= 3;
  if ( v18 == 3 )
    goto LABEL_35;
LABEL_19:
  if ( v19 )
  {
    if ( (unsigned int)(v18 - 1) > 1 )
      goto LABEL_26;
    goto LABEL_21;
  }
  if ( v18 != 4 )
    goto LABEL_26;
  if ( !*(_DWORD *)(v17 + 352) )
    goto LABEL_26;
  v35 = sub_22077B0(0x30u);
  v21 = v35;
  if ( !v35 )
    goto LABEL_26;
  sub_39ACB90(v35, a1);
LABEL_23:
  v72.m128i_i64[0] = v21;
  v72.m128i_i64[1] = (__int64)"write_exception";
  v73.m128i_i64[0] = (__int64)"DWARF Exception Writer";
  v73.m128i_i64[1] = (__int64)"dwarf";
  v74 = "DWARF Emission";
  v24 = *(unsigned int *)(a1 + 432);
  if ( (unsigned int)v24 >= *(_DWORD *)(a1 + 436) )
  {
    sub_16CD150(a1 + 424, (const void *)(a1 + 440), 0, 40, v22, v23);
    v24 = *(unsigned int *)(a1 + 432);
  }
  v25 = (__m128i *)(*(_QWORD *)(a1 + 424) + 40 * v24);
  *v25 = _mm_loadu_si128(&v72);
  v25[1] = _mm_loadu_si128(&v73);
  v25[2].m128i_i64[0] = (__int64)v74;
  ++*(_DWORD *)(a1 + 432);
LABEL_26:
  v26 = sub_16328F0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 1688LL), "cfguard", 7u);
  if ( v26 && *(_QWORD *)(v26 + 136) )
  {
    v27 = sub_22077B0(0x10u);
    v30 = v27;
    if ( v27 )
      sub_39AC1B0(v27, a1);
    v72.m128i_i64[0] = v30;
    v72.m128i_i64[1] = (__int64)"Control Flow Guard";
    v73.m128i_i64[0] = (__int64)"Control Flow Guard Tables";
    v73.m128i_i64[1] = (__int64)"dwarf";
    v74 = "DWARF Emission";
    v31 = *(unsigned int *)(a1 + 432);
    if ( (unsigned int)v31 >= *(_DWORD *)(a1 + 436) )
    {
      sub_16CD150(a1 + 424, (const void *)(a1 + 440), 0, 40, v28, v29);
      v31 = *(unsigned int *)(a1 + 432);
    }
    v32 = (__m128i *)(*(_QWORD *)(a1 + 424) + 40 * v31);
    *v32 = _mm_loadu_si128(&v72);
    v32[1] = _mm_loadu_si128(&v73);
    v32[2].m128i_i64[0] = (__int64)v74;
    ++*(_DWORD *)(a1 + 432);
  }
  return 0;
}
