// Function: sub_3275DB0
// Address: 0x3275db0
//
__int64 __fastcall sub_3275DB0(__int64 a1, __int64 a2, char a3)
{
  __int64 *v6; // rcx
  __int64 v7; // r10
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // r11
  unsigned int v11; // edx
  __int64 result; // rax
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // rax
  int v16; // r12d
  unsigned __int16 *v17; // rax
  unsigned __int16 v18; // di
  __int64 v19; // rax
  __int128 v20; // rax
  __int64 v21; // rsi
  int v22; // r13d
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rsi
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r9
  __int16 v30; // dx
  __int64 v31; // rax
  int v32; // esi
  __int128 v33; // rax
  __int64 v34; // rsi
  int v35; // r13d
  __int64 v36; // r12
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rsi
  __int64 v40; // r8
  __int64 v41; // rax
  __int64 v42; // r9
  __int16 v43; // dx
  __int64 v44; // rax
  int v45; // esi
  bool v46; // al
  bool v47; // al
  __int128 v48; // [rsp-20h] [rbp-D0h]
  __int128 v49; // [rsp-10h] [rbp-C0h]
  unsigned int v50; // [rsp+Ch] [rbp-A4h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __int128 v53; // [rsp+20h] [rbp-90h]
  __int128 v54; // [rsp+20h] [rbp-90h]
  __int128 v55; // [rsp+30h] [rbp-80h]
  __int128 v56; // [rsp+30h] [rbp-80h]
  __int64 v57; // [rsp+30h] [rbp-80h]
  __int64 v58; // [rsp+40h] [rbp-70h]
  __int64 v59; // [rsp+58h] [rbp-58h]
  __int64 v60; // [rsp+58h] [rbp-58h]
  __int64 v61; // [rsp+60h] [rbp-50h] BYREF
  int v62; // [rsp+68h] [rbp-48h]
  __int64 v63; // [rsp+70h] [rbp-40h] BYREF
  __int64 v64; // [rsp+78h] [rbp-38h]

  v6 = *(__int64 **)(a1 + 40);
  v7 = v6[1];
  v8 = *v6;
  v9 = *((unsigned int *)v6 + 2);
  v10 = v6[5];
  v11 = *((_DWORD *)v6 + 12);
  if ( a3 )
  {
    v9 = v11;
    v11 = *((_DWORD *)v6 + 2);
    v8 = v6[5];
    v10 = *v6;
  }
  if ( *(_DWORD *)(v10 + 24) != 206 )
    return 0;
  v13 = *(_QWORD *)(v10 + 56);
  if ( !v13 )
    return 0;
  v14 = 1;
  do
  {
    while ( v11 != *(_DWORD *)(v13 + 8) )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        goto LABEL_14;
    }
    if ( !v14 )
      return 0;
    v15 = *(_QWORD *)(v13 + 32);
    if ( !v15 )
      goto LABEL_15;
    if ( v11 == *(_DWORD *)(v15 + 8) )
      return 0;
    v13 = *(_QWORD *)(v15 + 32);
    v14 = 0;
  }
  while ( v13 );
LABEL_14:
  if ( v14 == 1 )
    return 0;
LABEL_15:
  v16 = *(_DWORD *)(a1 + 24);
  if ( v16 == 60 )
  {
    v57 = v6[1];
    v60 = v10;
    if ( !(unsigned __int8)sub_33DE9F0(a2, v6[5], v6[6], 0) )
      return 0;
    v16 = *(_DWORD *)(a1 + 24);
    v10 = v60;
    v7 = v57;
  }
  else if ( v16 > 62 )
  {
    if ( (unsigned int)(v16 - 65) <= 1 )
      return 0;
  }
  else if ( v16 > 58 )
  {
    return 0;
  }
  v17 = *(unsigned __int16 **)(a1 + 48);
  v52 = v7;
  v18 = *v17;
  v59 = *((_QWORD *)v17 + 1);
  v19 = *(_QWORD *)(v10 + 40);
  v51 = *(_QWORD *)v19;
  v55 = (__int128)_mm_loadu_si128((const __m128i *)(v19 + 40));
  v50 = *(_DWORD *)(v19 + 8);
  v53 = (__int128)_mm_loadu_si128((const __m128i *)(v19 + 80));
  if ( (unsigned __int8)sub_33E1910((unsigned int)v16, *(unsigned int *)(a1 + 28), v55, *((_QWORD *)&v55 + 1)) )
  {
    *(_QWORD *)&v33 = sub_33FB960(a2, v8, v9 | v52 & 0xFFFFFFFF00000000LL);
    v34 = *(_QWORD *)(a1 + 80);
    v35 = *(_DWORD *)(a1 + 28);
    v56 = v33;
    v63 = v34;
    if ( v34 )
      sub_B96E90((__int64)&v63, v34, 1);
    LODWORD(v64) = *(_DWORD *)(a1 + 72);
    v36 = sub_3405C90(a2, v16, (unsigned int)&v63, v18, v59, v35, v56, v53);
    v38 = v37;
    if ( v63 )
      sub_B91220((__int64)&v63, v63);
    v39 = *(_QWORD *)(a1 + 80);
    v61 = v39;
    if ( v39 )
      sub_B96E90((__int64)&v61, v39, 1);
    v62 = *(_DWORD *)(a1 + 72);
    v40 = v51;
    v41 = *(_QWORD *)(v51 + 48) + 16LL * v50;
    v42 = v50;
    v43 = *(_WORD *)v41;
    v44 = *(_QWORD *)(v41 + 8);
    LOWORD(v63) = v43;
    v64 = v44;
    if ( v43 )
    {
      v45 = ((unsigned __int16)(v43 - 17) < 0xD4u) + 205;
    }
    else
    {
      v46 = sub_30070B0((__int64)&v63);
      v40 = v51;
      v42 = v50;
      v45 = 205 - (!v46 - 1);
    }
    *((_QWORD *)&v49 + 1) = v38;
    *(_QWORD *)&v49 = v36;
    result = sub_340EC60(a2, v45, (unsigned int)&v61, v18, v59, 0, v40, v42, v56, v49);
    if ( v61 )
      goto LABEL_29;
    return result;
  }
  if ( !(unsigned __int8)sub_33E1910((unsigned int)v16, *(unsigned int *)(a1 + 28), v53, *((_QWORD *)&v53 + 1)) )
    return 0;
  *(_QWORD *)&v20 = sub_33FB960(a2, v8, v9 | v52 & 0xFFFFFFFF00000000LL);
  v21 = *(_QWORD *)(a1 + 80);
  v22 = *(_DWORD *)(a1 + 28);
  v54 = v20;
  v63 = v21;
  if ( v21 )
    sub_B96E90((__int64)&v63, v21, 1);
  LODWORD(v64) = *(_DWORD *)(a1 + 72);
  v23 = sub_3405C90(a2, v16, (unsigned int)&v63, v18, v59, v22, v54, v55);
  v25 = v24;
  if ( v63 )
    sub_B91220((__int64)&v63, v63);
  v26 = *(_QWORD *)(a1 + 80);
  v61 = v26;
  if ( v26 )
    sub_B96E90((__int64)&v61, v26, 1);
  v62 = *(_DWORD *)(a1 + 72);
  v27 = v51;
  v28 = *(_QWORD *)(v51 + 48) + 16LL * v50;
  v29 = v50;
  v30 = *(_WORD *)v28;
  v31 = *(_QWORD *)(v28 + 8);
  LOWORD(v63) = v30;
  v64 = v31;
  if ( v30 )
  {
    v32 = ((unsigned __int16)(v30 - 17) < 0xD4u) + 205;
  }
  else
  {
    v47 = sub_30070B0((__int64)&v63);
    v27 = v51;
    v29 = v50;
    v32 = 205 - (!v47 - 1);
  }
  *((_QWORD *)&v48 + 1) = v25;
  *(_QWORD *)&v48 = v23;
  result = sub_340EC60(a2, v32, (unsigned int)&v61, v18, v59, 0, v27, v29, v48, v54);
  if ( v61 )
  {
LABEL_29:
    v58 = result;
    sub_B91220((__int64)&v61, v61);
    return v58;
  }
  return result;
}
