// Function: sub_210D080
// Address: 0x210d080
//
void __fastcall sub_210D080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 **v10; // rax
  __int64 v11; // rdx
  __int64 **v12; // r14
  __int64 v13; // r12
  unsigned __int64 v14; // rax
  __int64 **v15; // rbx
  __int64 *v16; // r13
  int v17; // eax
  __int64 *v18; // rax
  const char *v19; // r15
  size_t v20; // rdx
  size_t v21; // r14
  size_t v22; // rdx
  const char *v23; // rdi
  size_t v24; // r12
  __int64 *v25; // r12
  const char *v26; // rax
  size_t v27; // rdx
  _BYTE *v28; // rdi
  char *v29; // rsi
  _BYTE *v30; // rax
  size_t v31; // r15
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  __m128i si128; // xmm0
  __int64 v36; // rdi
  __int64 (*v37)(); // rax
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  __int64 v40; // rax
  unsigned int v41; // r15d
  unsigned int v42; // r13d
  int v43; // eax
  __int64 v44; // rdx
  _BYTE *v45; // rax
  _BYTE *v46; // rax
  __int64 **v48; // [rsp+18h] [rbp-278h]
  __int64 v49; // [rsp+20h] [rbp-270h]
  __int64 **v50; // [rsp+28h] [rbp-268h]
  __int64 **v51; // [rsp+28h] [rbp-268h]
  __int64 v52[2]; // [rsp+30h] [rbp-260h] BYREF
  void (__fastcall *v53)(__int64 *, __int64 *, __int64); // [rsp+40h] [rbp-250h]
  void (__fastcall *v54)(__int64 *, __int64); // [rsp+48h] [rbp-248h]
  __int64 **v55; // [rsp+50h] [rbp-240h] BYREF
  __int64 v56; // [rsp+58h] [rbp-238h]
  _BYTE v57[560]; // [rsp+60h] [rbp-230h] BYREF

  v55 = (__int64 **)v57;
  v56 = 0x4000000000LL;
  if ( !*(_DWORD *)(a1 + 176) )
    goto LABEL_2;
  v6 = *(__int64 **)(a1 + 168);
  v7 = a2;
  v8 = &v6[4 * *(unsigned int *)(a1 + 184)];
  if ( v6 == v8 )
    goto LABEL_2;
  while ( 1 )
  {
    v9 = v6;
    if ( *v6 != -8 && *v6 != -16 )
      break;
    v6 += 4;
    if ( v8 == v6 )
      goto LABEL_2;
  }
  if ( v8 == v6 )
  {
LABEL_2:
    v48 = (__int64 **)v57;
    goto LABEL_3;
  }
  v10 = (__int64 **)v57;
  v11 = 0;
  while ( 1 )
  {
    v10[v11] = v9;
    v9 += 4;
    v11 = (unsigned int)(v56 + 1);
    LODWORD(v56) = v56 + 1;
    if ( v9 == v8 )
      break;
    while ( *v9 == -16 || *v9 == -8 )
    {
      v9 += 4;
      if ( v8 == v9 )
        goto LABEL_17;
    }
    if ( v8 == v9 )
      break;
    if ( HIDWORD(v56) <= (unsigned int)v11 )
    {
      sub_16CD150((__int64)&v55, v57, 0, 8, a5, a6);
      v11 = (unsigned int)v56;
    }
    v10 = v55;
  }
LABEL_17:
  v12 = v55;
  v13 = 8 * v11;
  v48 = &v55[v11];
  if ( v48 == v55 )
    goto LABEL_3;
  _BitScanReverse64(&v14, v13 >> 3);
  sub_210CB60(v55, v48, 2LL * (int)(63 - (v14 ^ 0x3F)));
  if ( (unsigned __int64)v13 > 0x80 )
  {
    v50 = v12 + 16;
    sub_210C560(v12, v12 + 16);
    if ( v48 == v12 + 16 )
      goto LABEL_32;
    while ( 1 )
    {
      v15 = v50;
      v16 = *v50;
      while ( 1 )
      {
        v19 = sub_1649960(**(v15 - 1));
        v21 = v20;
        v23 = sub_1649960(*v16);
        v24 = v22;
        if ( v22 > v21 )
          break;
        if ( v22 )
        {
          v17 = memcmp(v23, v19, v22);
          if ( v17 )
            goto LABEL_29;
        }
        if ( v24 == v21 )
          goto LABEL_30;
LABEL_24:
        if ( v24 >= v21 )
          goto LABEL_30;
LABEL_25:
        v18 = *--v15;
        v15[1] = v18;
      }
      if ( !v21 )
        goto LABEL_30;
      v17 = memcmp(v23, v19, v21);
      if ( !v17 )
        goto LABEL_24;
LABEL_29:
      if ( v17 < 0 )
        goto LABEL_25;
LABEL_30:
      ++v50;
      *v15 = v16;
      if ( v48 == v50 )
      {
        v7 = a2;
        goto LABEL_32;
      }
    }
  }
  sub_210C560(v12, v48);
LABEL_32:
  v48 = &v55[(unsigned int)v56];
  if ( v55 == v48 )
    goto LABEL_3;
  v51 = v55;
  do
  {
    v25 = *v51;
    v26 = sub_1649960(**v51);
    v28 = *(_BYTE **)(v7 + 24);
    v29 = (char *)v26;
    v30 = *(_BYTE **)(v7 + 16);
    v31 = v27;
    if ( v30 - v28 < v27 )
    {
      v32 = sub_16E7EE0(v7, v29, v27);
      v28 = *(_BYTE **)(v32 + 24);
      if ( v28 != *(_BYTE **)(v32 + 16) )
        goto LABEL_38;
    }
    else
    {
      v32 = v7;
      if ( v27 )
      {
        memcpy(v28, v29, v27);
        v30 = *(_BYTE **)(v7 + 16);
        v28 = (_BYTE *)(v31 + *(_QWORD *)(v7 + 24));
        *(_QWORD *)(v7 + 24) = v28;
      }
      if ( v28 != v30 )
      {
LABEL_38:
        *v28 = 32;
        v33 = *(_QWORD *)(v32 + 24) + 1LL;
        v34 = *(_QWORD *)(v32 + 16);
        *(_QWORD *)(v32 + 24) = v33;
        if ( (unsigned __int64)(v34 - v33) > 0x14 )
          goto LABEL_39;
        goto LABEL_62;
      }
    }
    v32 = sub_16E7EE0(v32, " ", 1u);
    v33 = *(_QWORD *)(v32 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v32 + 16) - v33) > 0x14 )
    {
LABEL_39:
      si128 = _mm_load_si128((const __m128i *)&xmmword_430B360);
      *(_DWORD *)(v33 + 16) = 980644453;
      *(_BYTE *)(v33 + 20) = 32;
      *(__m128i *)v33 = si128;
      *(_QWORD *)(v32 + 24) += 21LL;
      goto LABEL_40;
    }
LABEL_62:
    sub_16E7EE0(v32, "Clobbered Registers: ", 0x15u);
LABEL_40:
    v36 = *(_QWORD *)(a1 + 192);
    v37 = *(__int64 (**)())(*(_QWORD *)v36 + 16LL);
    if ( v37 == sub_16FF750 )
      BUG();
    v38 = ((__int64 (__fastcall *)(__int64, __int64))v37)(v36, *v25);
    v39 = *(__int64 (**)())(*(_QWORD *)v38 + 112LL);
    if ( v39 == sub_1D00B10 )
      BUG();
    v40 = ((__int64 (__fastcall *)(__int64))v39)(v38);
    v41 = 1;
    v42 = *(_DWORD *)(v40 + 16);
    v49 = v40;
    if ( v42 > 1 )
    {
      do
      {
        while ( 1 )
        {
          v43 = *(_DWORD *)(v25[1] + 4LL * (v41 >> 5));
          if ( !_bittest(&v43, v41) )
          {
            sub_1F4AA00(v52, v41, v49, 0, 0);
            if ( !v53 )
              sub_4263D6(v52, v41, v44);
            v54(v52, v7);
            v45 = *(_BYTE **)(v7 + 24);
            if ( *(_BYTE **)(v7 + 16) == v45 )
            {
              sub_16E7EE0(v7, " ", 1u);
            }
            else
            {
              *v45 = 32;
              ++*(_QWORD *)(v7 + 24);
            }
            if ( v53 )
              break;
          }
          if ( ++v41 == v42 )
            goto LABEL_51;
        }
        ++v41;
        v53(v52, v52, 3);
      }
      while ( v41 != v42 );
    }
LABEL_51:
    v46 = *(_BYTE **)(v7 + 24);
    if ( *(_BYTE **)(v7 + 16) == v46 )
    {
      sub_16E7EE0(v7, "\n", 1u);
    }
    else
    {
      *v46 = 10;
      ++*(_QWORD *)(v7 + 24);
    }
    ++v51;
  }
  while ( v48 != v51 );
  v48 = v55;
LABEL_3:
  if ( v48 != (__int64 **)v57 )
    _libc_free((unsigned __int64)v48);
}
