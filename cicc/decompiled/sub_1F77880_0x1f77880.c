// Function: sub_1F77880
// Address: 0x1f77880
//
__int64 *__fastcall sub_1F77880(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 *result; // rax
  __int16 v9; // dx
  char v12; // si
  __int64 *v13; // rdx
  __int64 v14; // r9
  int v15; // ecx
  __int64 v16; // rdi
  int v17; // ecx
  int v18; // ecx
  unsigned __int8 *v19; // rcx
  unsigned __int8 v20; // di
  const void **v21; // r8
  unsigned int v22; // r14d
  __int64 (*v23)(); // rcx
  __int64 v24; // rsi
  __int64 *v25; // r11
  __int64 *v26; // r10
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 *v30; // rcx
  __int64 *v31; // r11
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 *v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // r8
  unsigned __int64 v39; // r9
  __int64 v40; // rsi
  __int64 *v41; // r12
  int v42; // ecx
  __int64 v43; // rcx
  unsigned int v44; // r10d
  __int64 v45; // r11
  char *v46; // rcx
  __int64 *v47; // [rsp+10h] [rbp-80h]
  __int64 v48; // [rsp+10h] [rbp-80h]
  __int64 *v49; // [rsp+18h] [rbp-78h]
  __int64 v50; // [rsp+18h] [rbp-78h]
  __int64 v51; // [rsp+20h] [rbp-70h]
  __int64 *v52; // [rsp+20h] [rbp-70h]
  __int64 *v53; // [rsp+20h] [rbp-70h]
  __int64 v54; // [rsp+20h] [rbp-70h]
  unsigned __int64 v55; // [rsp+28h] [rbp-68h]
  unsigned __int64 v56; // [rsp+28h] [rbp-68h]
  __int128 *v57; // [rsp+30h] [rbp-60h]
  __int128 v58; // [rsp+30h] [rbp-60h]
  const void **v59; // [rsp+40h] [rbp-50h]
  __int64 *v60; // [rsp+40h] [rbp-50h]
  const void **v61; // [rsp+48h] [rbp-48h]
  __int64 *v62; // [rsp+48h] [rbp-48h]
  __int64 v63; // [rsp+50h] [rbp-40h] BYREF
  int v64; // [rsp+58h] [rbp-38h]

  v5 = *(__int64 **)(a2 + 32);
  v6 = *v5;
  v7 = *(_QWORD *)(*v5 + 48);
  if ( !v7 || *(_QWORD *)(v7 + 32) )
    return 0;
  v9 = *(_WORD *)(v6 + 24);
  if ( v9 == 118 )
  {
    v12 = 1;
  }
  else
  {
    if ( v9 > 118 )
    {
      if ( (unsigned __int16)(v9 - 119) > 1u )
        return 0;
    }
    else if ( v9 != 52 || *(_WORD *)(a2 + 24) != 122 )
    {
      return 0;
    }
    v12 = 0;
  }
  v13 = *(__int64 **)(v6 + 32);
  v14 = v13[5];
  v15 = *(unsigned __int16 *)(v14 + 24);
  if ( v15 != 10 && v15 != 32 || (*(_BYTE *)(v14 + 26) & 8) != 0 )
    return 0;
  v16 = *v13;
  v17 = *(unsigned __int16 *)(*v13 + 24);
  if ( v17 == 122 )
  {
    v42 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v16 + 32) + 40LL) + 24LL);
    if ( v42 != 32 && v42 != 10 )
      return 0;
  }
  else if ( v17 == 47 || v17 == 134 )
  {
    v43 = *(_QWORD *)(a2 + 48);
    if ( v43 && !*(_QWORD *)(v43 + 32) )
      return 0;
  }
  else
  {
    if ( (unsigned int)(v17 - 123) > 1 )
      return 0;
    v18 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v16 + 32) + 40LL) + 24LL);
    if ( v18 != 10 && v18 != 32 )
      return 0;
  }
  v19 = *(unsigned __int8 **)(a2 + 40);
  v20 = *v19;
  v21 = (const void **)*((_QWORD *)v19 + 1);
  v22 = *v19;
  v61 = v21;
  if ( *(_WORD *)(a2 + 24) == 123 )
  {
    v44 = *(_DWORD *)(*(_QWORD *)(v14 + 88) + 32LL);
    v45 = *(_QWORD *)(*(_QWORD *)(v14 + 88) + 24LL);
    if ( v44 > 0x40 )
      v45 = *(_QWORD *)(v45 + 8LL * ((v44 - 1) >> 6));
    if ( v12 != ((v45 & (1LL << ((unsigned __int8)v44 - 1))) != 0) )
      return 0;
  }
  v23 = *(__int64 (**)())(*a1[1] + 1120);
  if ( v23 != sub_1F6BB60 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64))v23)(a1[1], v6) )
    {
      v46 = *(char **)(a2 + 40);
      v13 = *(__int64 **)(v6 + 32);
      v5 = *(__int64 **)(a2 + 32);
      v20 = *v46;
      v21 = (const void **)*((_QWORD *)v46 + 1);
      v14 = v13[5];
      goto LABEL_21;
    }
    return 0;
  }
LABEL_21:
  v24 = *(_QWORD *)(v14 + 72);
  v25 = *a1;
  v57 = (__int128 *)(v5 + 5);
  v26 = v13 + 5;
  v27 = v20;
  v63 = v24;
  if ( v24 )
  {
    v47 = v13 + 5;
    v49 = v25;
    v59 = v21;
    v51 = v14;
    sub_1623A60((__int64)&v63, v24, 2);
    v27 = v20;
    v26 = v47;
    v25 = v49;
    v21 = v59;
    v14 = v51;
  }
  v28 = *(unsigned __int16 *)(a2 + 24);
  v64 = *(_DWORD *)(v14 + 64);
  *(_QWORD *)&v58 = sub_1D332F0(v25, v28, (__int64)&v63, v27, v21, 0, a3, a4, a5, *v26, v26[1], *v57);
  *((_QWORD *)&v58 + 1) = v29;
  if ( v63 )
    sub_161E7C0((__int64)&v63, v63);
  v30 = *(__int64 **)(v6 + 32);
  v31 = *a1;
  v32 = *(_QWORD *)(a2 + 32);
  v33 = *v30;
  v34 = *(_QWORD *)(*v30 + 72);
  v63 = v34;
  if ( v34 )
  {
    v48 = v32;
    v50 = v33;
    v60 = v30;
    v52 = v31;
    sub_1623A60((__int64)&v63, v34, 2);
    v32 = v48;
    v33 = v50;
    v30 = v60;
    v31 = v52;
  }
  v35 = *(unsigned __int16 *)(a2 + 24);
  v64 = *(_DWORD *)(v33 + 64);
  v36 = sub_1D332F0(v31, v35, (__int64)&v63, v22, v61, 0, a3, a4, a5, *v30, v30[1], *(_OWORD *)(v32 + 40));
  v38 = (__int64)v36;
  v39 = v37;
  if ( v63 )
  {
    v53 = v36;
    v55 = v37;
    sub_161E7C0((__int64)&v63, v63);
    v38 = (__int64)v53;
    v39 = v55;
  }
  v40 = *(_QWORD *)(a2 + 72);
  v41 = *a1;
  v63 = v40;
  if ( v40 )
  {
    v54 = v38;
    v56 = v39;
    sub_1623A60((__int64)&v63, v40, 2);
    v38 = v54;
    v39 = v56;
  }
  v64 = *(_DWORD *)(a2 + 64);
  result = sub_1D332F0(v41, *(unsigned __int16 *)(v6 + 24), (__int64)&v63, v22, v61, 0, a3, a4, a5, v38, v39, v58);
  if ( v63 )
  {
    v62 = result;
    sub_161E7C0((__int64)&v63, v63);
    return v62;
  }
  return result;
}
