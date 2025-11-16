// Function: sub_3280360
// Address: 0x3280360
//
void __fastcall sub_3280360(__int64 ***a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int16 v4; // ax
  __int64 **v5; // r13
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r8
  __int64 v14; // rcx
  __int16 v15; // r14
  bool v16; // al
  int *v17; // rdx
  bool v18; // si
  int v19; // eax
  __int64 *v20; // rdx
  __int64 v21; // rax
  int v22; // edx
  int v23; // eax
  __m128i v24; // xmm1
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __int64 *v27; // rdi
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 *v34; // rcx
  __int64 v35; // rdi
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // rax
  _QWORD *v39; // rax
  bool v40; // al
  int v41; // eax
  unsigned __int16 *v42; // rax
  int v43; // eax
  int v44; // ecx
  __int16 v45; // ax
  __int64 v46; // rsi
  __int64 v47; // [rsp+0h] [rbp-D0h]
  __int64 v48; // [rsp+0h] [rbp-D0h]
  __int64 v49; // [rsp+8h] [rbp-C8h]
  __int64 v50; // [rsp+8h] [rbp-C8h]
  __int64 v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+8h] [rbp-C8h]
  char v54[8]; // [rsp+18h] [rbp-B8h] BYREF
  __m128i v55; // [rsp+20h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+30h] [rbp-A0h]
  __m128i v57; // [rsp+40h] [rbp-90h]
  __int64 v58; // [rsp+50h] [rbp-80h]
  __m128i v59; // [rsp+60h] [rbp-70h] BYREF
  __m128i v60; // [rsp+70h] [rbp-60h] BYREF
  __m128i v61; // [rsp+80h] [rbp-50h] BYREF
  char v62; // [rsp+90h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( -858993459 * (unsigned int)((a2 - *(_QWORD *)(v2 + 40)) >> 3) )
    return;
  if ( *(_DWORD *)(v2 + 24) != 299 )
    return;
  v3 = *(_QWORD *)(v2 + 112);
  v58 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  if ( (*(_BYTE *)(v3 + 37) & 0xF) != 0 )
    return;
  v4 = *(_WORD *)(v2 + 32);
  v5 = *a1;
  if ( (v4 & 8) != 0 )
    return;
  if ( (v4 & 0x380) != 0 )
    return;
  v7 = **v5;
  if ( ((*(_BYTE *)(v7 + 32) & 0x10) != 0) != ((*(_BYTE *)(v2 + 32) & 0x10) != 0) )
    return;
  v8 = v5[1][1];
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 360LL);
  if ( v9 != sub_2FE3000 && !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v9)(v8, v7, v2) )
    return;
  v10 = sub_33CF5B0(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v2 + 40) + 48LL));
  v12 = (__int64)v5[2];
  v13 = v10;
  v14 = v11;
  v15 = *(_WORD *)v12;
  if ( *(_WORD *)v12 )
  {
    if ( (unsigned __int16)(v15 - 2) <= 7u
      || (unsigned __int16)(v15 - 17) <= 0x6Cu
      || (unsigned __int16)(v15 - 176) <= 0x1Fu )
    {
      goto LABEL_12;
    }
  }
  else
  {
    v48 = v11;
    v51 = v10;
    v40 = sub_3007070((__int64)v5[2]);
    v13 = v51;
    v14 = v48;
    if ( v40 )
    {
LABEL_12:
      v47 = v14;
      v49 = v13;
      v16 = sub_3280240(v12, *(unsigned __int16 *)(v2 + 96), *(_QWORD *)(v2 + 104));
      v17 = (int *)v5[3];
      v13 = v49;
      v14 = v47;
      v18 = !v16;
      goto LABEL_13;
    }
  }
  v17 = (int *)v5[3];
  v41 = *v17;
  if ( *(_WORD *)(v2 + 96) != v15 )
  {
    if ( v41 != 2 )
    {
      if ( v41 == 3 || v41 == 1 )
        return;
      goto LABEL_79;
    }
    goto LABEL_51;
  }
  if ( v15 )
  {
    if ( v41 != 2 )
    {
      if ( v41 == 3 )
        goto LABEL_16;
      if ( v41 == 1 )
        goto LABEL_29;
LABEL_79:
      BUG();
    }
LABEL_51:
    if ( (*(_BYTE *)(v2 + 33) & 4) != 0 )
      return;
    v52 = v13;
    v42 = (unsigned __int16 *)(*(_QWORD *)(v13 + 48) + 16LL * (unsigned int)v14);
    if ( !sub_3280240(v12, *v42, *((_QWORD *)v42 + 1)) )
      return;
    v43 = *(_DWORD *)(v52 + 24);
    if ( v43 != 158 && v43 != 161 )
      return;
    goto LABEL_31;
  }
  v18 = *(_QWORD *)(v2 + 104) != *(_QWORD *)(v12 + 8);
LABEL_13:
  v19 = *v17;
  if ( *v17 == 2 )
    goto LABEL_51;
  if ( v19 == 3 )
  {
    if ( v18 )
      return;
LABEL_16:
    if ( *(_DWORD *)(v13 + 24) != 298 )
      return;
    v50 = v13;
    sub_33644B0(&v59, v13, *v5[1], v14);
    v20 = v5[4];
    if ( *(_WORD *)(v50 + 96) != *(_WORD *)v20 || !*(_WORD *)v20 && *(_QWORD *)(v50 + 104) != v20[1] )
      return;
    v21 = *(_QWORD *)(v50 + 56);
    if ( !v21 )
      return;
    v22 = 1;
    do
    {
      if ( !*(_DWORD *)(v21 + 8) )
      {
        if ( !v22 )
          return;
        v21 = *(_QWORD *)(v21 + 32);
        if ( !v21 )
          goto LABEL_70;
        if ( !*(_DWORD *)(v21 + 8) )
          return;
        v22 = 0;
      }
      v21 = *(_QWORD *)(v21 + 32);
    }
    while ( v21 );
    if ( v22 == 1 )
      return;
LABEL_70:
    if ( (*(_BYTE *)(*(_QWORD *)(v50 + 112) + 37LL) & 0xF) != 0 )
      return;
    v45 = *(_WORD *)(v50 + 32);
    if ( (v45 & 8) != 0 )
      return;
    if ( (v45 & 0x380) != 0 )
      return;
    v46 = *v5[5];
    if ( ((*(_BYTE *)(v46 + 32) & 0x10) != 0) != ((*(_BYTE *)(v50 + 32) & 0x10) != 0)
      || !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5[1][1] + 360LL))(
            v5[1][1],
            v46,
            v50)
      || !(unsigned __int8)sub_3364290(v5[6], &v59, *v5[1], v54) )
    {
      return;
    }
    goto LABEL_31;
  }
  if ( v19 != 1 )
    goto LABEL_79;
  if ( v18 )
    return;
LABEL_29:
  v23 = *(_DWORD *)(v13 + 24);
  if ( v23 > 12 )
  {
    if ( v23 != 156 )
      return;
    v53 = v13;
    if ( !(unsigned __int8)sub_33CA6D0(v13) && !(unsigned __int8)sub_33CA720(v53) )
      return;
  }
  else if ( v23 <= 10 )
  {
    return;
  }
LABEL_31:
  sub_33644B0(&v59, v2, *v5[1], v14);
  v24 = _mm_load_si128(&v59);
  v25 = _mm_load_si128(&v60);
  v26 = _mm_load_si128(&v61);
  LOBYTE(v58) = v62;
  v56 = v25;
  v57 = v26;
  v27 = v5[7];
  v55 = v24;
  if ( !(unsigned __int8)sub_3364290(v27, &v55, *v5[1], &v59) )
    return;
  v30 = (__int64)*a1[1];
  v31 = *(_QWORD *)(v30 + 880);
  v32 = *(unsigned int *)(v30 + 896);
  if ( (_DWORD)v32 )
  {
    v28 = (unsigned int)(v32 - 1);
    v33 = v28 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v34 = (__int64 *)(v31 + 24LL * v33);
    v35 = *v34;
    if ( v2 == *v34 )
    {
LABEL_34:
      if ( v34 != (__int64 *)(v31 + 24 * v32)
        && *a1[2] == (__int64 *)v34[1]
        && *((_DWORD *)v34 + 4) > (unsigned int)qword_5037FA8 )
      {
        return;
      }
    }
    else
    {
      v44 = 1;
      while ( v35 != -4096 )
      {
        v29 = (unsigned int)(v44 + 1);
        v33 = v28 & (v44 + v33);
        v34 = (__int64 *)(v31 + 24LL * v33);
        v35 = *v34;
        if ( v2 == *v34 )
          goto LABEL_34;
        v44 = v29;
      }
    }
  }
  v36 = (__int64)a1[3];
  v37 = v59.m128i_i64[0];
  v38 = *(unsigned int *)(v36 + 8);
  if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v36 + 12) )
  {
    sub_C8D5F0(v36, (const void *)(v36 + 16), v38 + 1, 0x10u, v28, v29);
    v38 = *(unsigned int *)(v36 + 8);
  }
  v39 = (_QWORD *)(*(_QWORD *)v36 + 16 * v38);
  *v39 = v2;
  v39[1] = v37;
  ++*(_DWORD *)(v36 + 8);
}
