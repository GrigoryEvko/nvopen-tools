// Function: sub_38C9E40
// Address: 0x38c9e40
//
__int64 __fastcall sub_38C9E40(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rcx
  void (__fastcall *v8)(__int64 *, __int64, _QWORD); // r8
  __int64 v9; // rsi
  __int64 v10; // r15
  _QWORD *v11; // rbx
  unsigned int v12; // r12d
  int v13; // r13d
  int v14; // esi
  int v15; // r15d
  __int64 *v16; // r15
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 v24; // r12
  unsigned __int16 v25; // ax
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // r12d
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rsi
  void (__fastcall *v38)(__int64 *, char *); // r13
  char *v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rdx
  const char *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // r15
  __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // r15
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // rbx
  unsigned int v55; // r12d
  __int64 *v56; // r13
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // r15
  __int64 v61; // rax
  __int64 v62; // r15
  __int64 v63; // [rsp+8h] [rbp-78h]
  __int64 v64; // [rsp+10h] [rbp-70h]
  __int64 v65; // [rsp+18h] [rbp-68h]
  char v66; // [rsp+20h] [rbp-60h]
  __int64 *v67; // [rsp+28h] [rbp-58h]
  __int64 *v68; // [rsp+28h] [rbp-58h]
  __int64 v69; // [rsp+30h] [rbp-50h]
  __int64 v70; // [rsp+30h] [rbp-50h]
  __int64 v71; // [rsp+38h] [rbp-48h]
  __int64 v72; // [rsp+38h] [rbp-48h]
  __int64 v73; // [rsp+38h] [rbp-48h]
  __int64 v74; // [rsp+38h] [rbp-48h]
  __int64 v75; // [rsp+38h] [rbp-48h]
  int v76; // [rsp+44h] [rbp-3Ch] BYREF
  unsigned int *v77[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1[1];
  v66 = *(_BYTE *)(*(_QWORD *)(v2 + 16) + 356LL);
  if ( v66 )
  {
    v3 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 704))(a1, 0);
    v4 = a1[1];
    v64 = v3;
  }
  else
  {
    v64 = 0;
    v4 = a1[1];
  }
  sub_38BDCD0(v4, (__int64)a1);
  v5 = a1[1];
  result = *(_QWORD *)(v5 + 1088);
  v7 = *(_QWORD *)(v5 + 1080);
  if ( v7 == result )
    return result;
  v8 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 160);
  v9 = *(_QWORD *)(*(_QWORD *)(v2 + 32) + 80LL);
  if ( (unsigned __int64)(result - v7) <= 8 || *(_WORD *)(v5 + 1160) <= 2u )
  {
    v8(a1, v9, 0);
    if ( !v66 )
    {
      v10 = 0;
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 32) + 72LL), 0);
      v65 = 0;
      v63 = 0;
      goto LABEL_7;
    }
    v66 = 0;
  }
  else
  {
    v8(a1, v9, 0);
    v66 = 1;
  }
  v10 = sub_38BFA60(v2, 1);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v10, 0);
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 32) + 72LL), 0);
  v65 = sub_38BFA60(v2, 1);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v65, 0);
  if ( v66 )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 32) + 152LL), 0);
    v63 = sub_38BFA60(v2, 1);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v63, 0);
  }
  else
  {
    v63 = 0;
  }
LABEL_7:
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 32) + 144LL), 0);
  v11 = (_QWORD *)a1[1];
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(v11[4] + 144LL), 0);
  v12 = *(_DWORD *)(v11[2] + 8LL);
  v13 = 2 * v12 - ((2 * (_BYTE)v12 - 1) & 0xC);
  v14 = v13 + 12;
  if ( 2 * v12 == v13 )
  {
    v14 = 12;
    v13 = 0;
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(
    a1,
    (int)(2 * v12 + v14 + 2 * v12 * ((__int64)(v11[136] - v11[135]) >> 3) - 4),
    4);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 424))(a1, 2, 2);
  if ( v10 )
    sub_38DDC80(a1, v10, 4);
  else
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 4);
  v15 = 0;
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, (int)v12, 1);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  if ( v13 > 0 )
  {
    do
    {
      ++v15;
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
    }
    while ( v13 != v15 );
  }
  v16 = (__int64 *)v11[135];
  v67 = (__int64 *)v11[136];
  while ( v67 != v16 )
  {
    v17 = *v16++;
    v18 = *(_QWORD *)(v17 + 8);
    v71 = sub_38D7790(v17, v11);
    v69 = sub_38CF310(v18, 0, v11, 0);
    v72 = sub_38CF310(v71, 0, a1[1], 0);
    v19 = sub_38CF310(v18, 0, a1[1], 0);
    v20 = sub_38CB1F0(17, v72, v19, a1[1], 0);
    v21 = sub_38CB470(0, a1[1]);
    v22 = sub_38CB1F0(17, v20, v21, a1[1], 0);
    sub_38DDD30(a1, v69, v12, 0);
    sub_38C4F40(a1, v22, v12);
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 424))(a1, 0, v12);
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 424))(a1, 0, v12);
  if ( v66 )
  {
    v54 = (_QWORD *)a1[1];
    v55 = *(_DWORD *)(v54[2] + 8LL);
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(v54[4] + 152LL), 0);
    v68 = (__int64 *)v54[136];
    v56 = (__int64 *)v54[135];
    while ( v68 != v56 )
    {
      v57 = *v56++;
      v58 = *(_QWORD *)(v57 + 8);
      v70 = sub_38D7790(v57, v54);
      v74 = sub_38CF310(v58, 0, v54, 0);
      sub_38DD0A0(a1, (int)v55, 255);
      sub_38DDD30(a1, v74, v55, 0);
      v75 = sub_38CF310(v70, 0, a1[1], 0);
      v59 = sub_38CF310(v58, 0, a1[1], 0);
      v60 = sub_38CB1F0(17, v75, v59, a1[1], 0);
      v61 = sub_38CB470(0, a1[1]);
      v62 = sub_38CB1F0(17, v60, v61, a1[1], 0);
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 424))(a1, 0, v55);
      sub_38C4F40(a1, v62, v55);
    }
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 424))(a1, 0, v55);
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 424))(a1, 0, v55);
  }
  v23 = a1[1];
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v23 + 32) + 72LL), 0);
  sub_38DCDD0(a1, 1);
  sub_38DCDD0(a1, 17);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 424))(a1, 1, 1);
  v24 = -(__int64)(*(_WORD *)(v23 + 1160) < 4u);
  sub_38DCDD0(a1, 16);
  sub_38DCDD0(a1, (v24 & 0xFFFFFFFFFFFFFFEFLL) + 23);
  if ( *(_QWORD *)(v23 + 1088) - *(_QWORD *)(v23 + 1080) <= 8u || (v25 = *(_WORD *)(v23 + 1160), v25 <= 2u) )
  {
    sub_38DCDD0(a1, 17);
    sub_38DCDD0(a1, 1);
    sub_38DCDD0(a1, 18);
    sub_38DCDD0(a1, 1);
  }
  else
  {
    v26 = 23;
    if ( v25 == 3 )
      v26 = 6;
    sub_38DCDD0(a1, 85);
    sub_38DCDD0(a1, v26);
  }
  sub_38DCDD0(a1, 3);
  sub_38DCDD0(a1, 8);
  if ( *(_DWORD *)(v23 + 760) )
  {
    sub_38DCDD0(a1, 27);
    sub_38DCDD0(a1, 8);
  }
  if ( *(_QWORD *)(v23 + 1136) )
  {
    sub_38DCDD0(a1, 16354);
    sub_38DCDD0(a1, 8);
  }
  sub_38DCDD0(a1, 37);
  sub_38DCDD0(a1, 8);
  sub_38DCDD0(a1, 19);
  sub_38DCDD0(a1, 5);
  sub_38DCDD0(a1, 0);
  sub_38DCDD0(a1, 0);
  sub_38DCDD0(a1, 2);
  sub_38DCDD0(a1, 10);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 424))(a1, 1, 1);
  sub_38DCDD0(a1, 3);
  sub_38DCDD0(a1, 8);
  sub_38DCDD0(a1, 58);
  sub_38DCDD0(a1, 6);
  sub_38DCDD0(a1, 59);
  sub_38DCDD0(a1, 6);
  sub_38DCDD0(a1, 17);
  sub_38DCDD0(a1, 1);
  sub_38DCDD0(a1, 39);
  sub_38DCDD0(a1, 12);
  sub_38DCDD0(a1, 0);
  sub_38DCDD0(a1, 0);
  sub_38DCDD0(a1, 3);
  sub_38DCDD0(a1, 24);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  sub_38DCDD0(a1, 0);
  sub_38DCDD0(a1, 0);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  v27 = a1[1];
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 160))(a1, *(_QWORD *)(*(_QWORD *)(v27 + 32) + 80LL), 0);
  v28 = sub_38BFA60(v27, 1);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v28, 0);
  v73 = sub_38BFA60(v27, 1);
  v29 = sub_38CF310(v73, 0, a1[1], 0);
  v30 = sub_38CF310(v28, 0, a1[1], 0);
  v31 = sub_38CB1F0(17, v29, v30, a1[1], 0);
  v32 = sub_38CB470(4, a1[1]);
  v33 = sub_38CB1F0(17, v31, v32, a1[1], 0);
  sub_38C4F40(a1, v33, 4u);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, *(unsigned __int16 *)(v27 + 1160), 2);
  v34 = *(_DWORD *)(*(_QWORD *)(v27 + 16) + 8LL);
  if ( *(_WORD *)(v27 + 1160) > 4u )
  {
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 424))(a1, 1, 1);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, (int)v34, 1);
  }
  if ( v65 )
    sub_38DDC80(a1, v65, 4);
  else
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 4);
  if ( *(_WORD *)(v27 + 1160) <= 4u )
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, (int)v34, 1);
  sub_38DCDD0(a1, 1);
  if ( v64 )
  {
    sub_38DDC80(a1, v64, 4);
    v35 = v63;
    if ( v63 )
    {
LABEL_33:
      sub_38DDC80(a1, v35, 4);
      goto LABEL_34;
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 4);
    v35 = v63;
    if ( v63 )
      goto LABEL_33;
  }
  v49 = *(_QWORD **)(v27 + 1080);
  v50 = *(_QWORD *)(*v49 + 8LL);
  v51 = sub_38D7790(*v49, v27);
  v52 = sub_38CF310(v50, 0, v27, 0);
  sub_38DDD30(a1, v52, v34, 0);
  v53 = sub_38CF310(v51, 0, v27, 0);
  sub_38DDD30(a1, v53, v34, 0);
LABEL_34:
  v76 = 0;
  v36 = *(_QWORD *)(v27 + 992);
  if ( !v36 )
  {
    v37 = v27 + 984;
LABEL_38:
    v77[0] = (unsigned int *)&v76;
    v37 = sub_38C9BD0((_QWORD *)(v27 + 976), v37, v77);
    goto LABEL_39;
  }
  do
  {
    v37 = v36;
    v36 = *(_QWORD *)(v36 + 16);
  }
  while ( v36 );
  if ( v27 + 984 == v37 || *(_DWORD *)(v37 + 32) )
    goto LABEL_38;
LABEL_39:
  if ( *(_DWORD *)(v37 + 56) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 400))(
      a1,
      **(_QWORD **)(v37 + 48),
      *(_QWORD *)(*(_QWORD *)(v37 + 48) + 8LL));
    v38 = *(void (__fastcall **)(__int64 *, char *))(*a1 + 400);
    v39 = sub_16C44C0(2);
    v38(a1, v39);
  }
  v40 = a1[1];
  v76 = 0;
  v41 = *(_QWORD *)(v40 + 992);
  if ( v41 )
  {
    do
    {
      v42 = v41;
      v41 = *(_QWORD *)(v41 + 16);
    }
    while ( v41 );
    if ( v40 + 984 != v42 && !*(_DWORD *)(v42 + 32) )
      goto LABEL_46;
  }
  else
  {
    v42 = v40 + 984;
  }
  v77[0] = (unsigned int *)&v76;
  v42 = sub_38C9BD0((_QWORD *)(v40 + 976), v42, v77);
LABEL_46:
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 400))(
    a1,
    *(_QWORD *)(*(_QWORD *)(v42 + 160) + 72LL),
    *(_QWORD *)(*(_QWORD *)(v42 + 160) + 80LL));
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  if ( *(_DWORD *)(v27 + 760) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD))(*a1 + 400))(a1, *(_QWORD *)(v27 + 752));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  }
  if ( *(_QWORD *)(v27 + 1136) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD))(*a1 + 400))(a1, *(_QWORD *)(v27 + 1128));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  }
  v43 = *(_QWORD *)(v27 + 1152);
  v44 = *(const char **)(v27 + 1144);
  if ( !v43 )
  {
    v44 = "llvm-mc (based on LLVM 7.0.1)";
    v43 = 29;
  }
  (*(void (__fastcall **)(__int64 *, const char *, __int64))(*a1 + 400))(a1, v44, v43);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 424))(a1, 32769, 2);
  v45 = a1[1];
  v46 = *(_QWORD *)(v45 + 1112);
  v47 = *(_QWORD *)(v45 + 1104);
  while ( v46 != v47 )
  {
    v47 += 32;
    sub_38DCDD0(a1, 2);
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 400))(a1, *(_QWORD *)(v47 - 32), *(_QWORD *)(v47 - 24));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, *(unsigned int *)(v47 - 16), 4);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, *(unsigned int *)(v47 - 12), 4);
    v48 = sub_38CF310(*(_QWORD *)(v47 - 8), 0, v27, 0);
    sub_38DDD30(a1, v48, v34, 0);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
    sub_38DCDD0(a1, 3);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 424))(a1, 0, 1);
  return (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v73, 0);
}
