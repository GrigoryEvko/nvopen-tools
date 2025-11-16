// Function: sub_2D6DEE0
// Address: 0x2d6dee0
//
__int64 __fastcall sub_2D6DEE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  char *v7; // rdx
  unsigned int v9; // eax
  _QWORD *v10; // r12
  char *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r8
  int v15; // r9d
  unsigned int v16; // edi
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v20; // rbx
  __int64 i; // rax
  __int64 v22; // r10
  unsigned int v23; // esi
  __int64 v24; // rdi
  unsigned int v25; // edx
  __int64 *v26; // rax
  _QWORD *v27; // rcx
  char v28; // al
  int v29; // eax
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  _QWORD *v33; // rax
  _QWORD *v34; // r8
  unsigned int v35; // eax
  _QWORD *v36; // rsi
  _QWORD *v37; // rdx
  _QWORD *v38; // rax
  __int64 v39; // rcx
  int v40; // edx
  int v41; // r11d
  __int64 *v42; // rax
  int v43; // r11d
  __int64 *v44; // r10
  int v45; // ebx
  int v46; // ecx
  __int64 v47; // rdi
  int v48; // eax
  int v49; // r9d
  __int64 v51; // [rsp+10h] [rbp-290h]
  __int64 v52; // [rsp+18h] [rbp-288h]
  __int64 v53; // [rsp+18h] [rbp-288h]
  __int64 *v54; // [rsp+28h] [rbp-278h] BYREF
  _QWORD *v55; // [rsp+30h] [rbp-270h] BYREF
  __int64 v56; // [rsp+38h] [rbp-268h]
  char *v57; // [rsp+40h] [rbp-260h] BYREF
  __int64 v58; // [rsp+48h] [rbp-258h]
  _QWORD v59[32]; // [rsp+50h] [rbp-250h] BYREF
  __int64 v60; // [rsp+150h] [rbp-150h] BYREF
  char *v61; // [rsp+158h] [rbp-148h]
  __int64 v62; // [rsp+160h] [rbp-140h]
  int v63; // [rsp+168h] [rbp-138h]
  unsigned __int8 v64; // [rsp+16Ch] [rbp-134h]
  char v65; // [rsp+170h] [rbp-130h] BYREF

  v6 = 1;
  v7 = (char *)v59;
  v57 = (char *)v59;
  v60 = 0;
  v62 = 32;
  v63 = 0;
  v64 = 1;
  v59[0] = a2;
  v61 = &v65;
  v58 = 0x2000000001LL;
  v9 = 1;
  while ( 1 )
  {
    v10 = *(_QWORD **)&v7[8 * v9 - 8];
    LODWORD(v58) = v9 - 1;
    if ( !(_BYTE)v6 )
      break;
    v11 = v61;
    v7 = &v61[8 * HIDWORD(v62)];
    if ( v61 == v7 )
    {
LABEL_23:
      if ( HIDWORD(v62) < (unsigned int)v62 )
      {
        ++HIDWORD(v62);
        *(_QWORD *)v7 = v10;
        v6 = v64;
        ++v60;
        goto LABEL_10;
      }
      break;
    }
    while ( v10 != *(_QWORD **)v11 )
    {
      v11 += 8;
      if ( v7 == v11 )
        goto LABEL_23;
    }
LABEL_7:
    v9 = v58;
    if ( !(_DWORD)v58 )
      goto LABEL_13;
LABEL_8:
    v7 = v57;
  }
  sub_C8CC70((__int64)&v60, (__int64)v10, (__int64)v7, v6, a5, a6);
  v6 = v64;
  if ( !(_BYTE)v7 )
    goto LABEL_7;
LABEL_10:
  if ( *(_BYTE *)v10 <= 0x1Cu )
    goto LABEL_7;
  a5 = sub_1020E10((__int64)v10, *(const __m128i **)(a1 + 32), v7, v6, a5, a6);
  if ( !a5 )
    goto LABEL_12;
  v20 = v10[2];
  for ( i = (unsigned int)v58; v20; v20 = *(_QWORD *)(v20 + 8) )
  {
    v22 = *(_QWORD *)(v20 + 24);
    if ( i + 1 > (unsigned __int64)HIDWORD(v58) )
    {
      v51 = *(_QWORD *)(v20 + 24);
      v52 = a5;
      sub_C8D5F0((__int64)&v57, v59, i + 1, 8u, a5, a6);
      i = (unsigned int)v58;
      v22 = v51;
      a5 = v52;
    }
    *(_QWORD *)&v57[8 * i] = v22;
    i = (unsigned int)(v58 + 1);
    LODWORD(v58) = v58 + 1;
  }
  v23 = *(_DWORD *)(a1 + 24);
  v55 = v10;
  v56 = a5;
  if ( !v23 )
  {
    ++*(_QWORD *)a1;
    v54 = 0;
LABEL_65:
    v53 = a5;
    v23 *= 2;
LABEL_66:
    sub_FAA400(a1, v23);
    sub_F9D990(a1, (__int64 *)&v55, &v54);
    v47 = (__int64)v55;
    a5 = v53;
    v46 = *(_DWORD *)(a1 + 16) + 1;
    v26 = v54;
    goto LABEL_57;
  }
  v24 = *(_QWORD *)(a1 + 8);
  v25 = (v23 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = (_QWORD *)*v26;
  if ( v10 == (_QWORD *)*v26 )
    goto LABEL_31;
  v43 = 1;
  v44 = 0;
  while ( v27 != (_QWORD *)-4096LL )
  {
    if ( v27 == (_QWORD *)-8192LL && !v44 )
      v44 = v26;
    v25 = (v23 - 1) & (v43 + v25);
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = (_QWORD *)*v26;
    if ( v10 == (_QWORD *)*v26 )
      goto LABEL_31;
    ++v43;
  }
  v45 = *(_DWORD *)(a1 + 16);
  if ( v44 )
    v26 = v44;
  ++*(_QWORD *)a1;
  v46 = v45 + 1;
  v54 = v26;
  if ( 4 * (v45 + 1) >= 3 * v23 )
    goto LABEL_65;
  v47 = (__int64)v10;
  if ( v23 - *(_DWORD *)(a1 + 20) - v46 <= v23 >> 3 )
  {
    v53 = a5;
    goto LABEL_66;
  }
LABEL_57:
  *(_DWORD *)(a1 + 16) = v46;
  if ( *v26 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v26 = v47;
  v26[1] = v56;
LABEL_31:
  sub_BD84D0((__int64)v10, a5);
  v28 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 != 84 )
  {
LABEL_36:
    if ( v28 != 86 )
      goto LABEL_43;
    if ( *(_BYTE *)(a1 + 876) )
    {
      v36 = *(_QWORD **)(a1 + 856);
      v37 = &v36[*(unsigned int *)(a1 + 868)];
      if ( v36 != v37 )
      {
        v38 = *(_QWORD **)(a1 + 856);
        while ( v10 != (_QWORD *)*v38 )
        {
          if ( v37 == ++v38 )
            goto LABEL_43;
        }
        v39 = (unsigned int)(*(_DWORD *)(a1 + 868) - 1);
        *(_DWORD *)(a1 + 868) = v39;
        *v38 = v36[v39];
        ++*(_QWORD *)(a1 + 848);
      }
      goto LABEL_43;
    }
    v42 = sub_C8CA60(a1 + 848, (__int64)v10);
    if ( !v42 )
      goto LABEL_43;
    *v42 = -2;
    ++*(_DWORD *)(a1 + 872);
    ++*(_QWORD *)(a1 + 848);
    sub_B43D60(v10);
    goto LABEL_12;
  }
  if ( (*(_BYTE *)(a1 + 320) & 1) != 0 )
  {
    v30 = a1 + 328;
    v31 = 31;
LABEL_34:
    v32 = v31 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v33 = (_QWORD *)(v30 + 16LL * v32);
    v34 = (_QWORD *)*v33;
    if ( v10 != (_QWORD *)*v33 )
    {
      v48 = 1;
      while ( v34 != (_QWORD *)-4096LL )
      {
        v49 = v48 + 1;
        v32 = v31 & (v48 + v32);
        v33 = (_QWORD *)(v30 + 16LL * v32);
        v34 = (_QWORD *)*v33;
        if ( v10 == (_QWORD *)*v33 )
          goto LABEL_35;
        v48 = v49;
      }
      goto LABEL_43;
    }
LABEL_35:
    *v33 = -8192;
    v35 = *(_DWORD *)(a1 + 320);
    ++*(_DWORD *)(a1 + 324);
    *(_DWORD *)(a1 + 320) = (2 * (v35 >> 1) - 2) | v35 & 1;
    sub_2D579F0(a1 + 40, (unsigned __int64 *)(a1 + 840));
    v28 = *(_BYTE *)v10;
    goto LABEL_36;
  }
  v29 = *(_DWORD *)(a1 + 336);
  v30 = *(_QWORD *)(a1 + 328);
  v31 = v29 - 1;
  if ( v29 )
    goto LABEL_34;
LABEL_43:
  sub_B43D60(v10);
LABEL_12:
  v9 = v58;
  v6 = v64;
  if ( (_DWORD)v58 )
    goto LABEL_8;
LABEL_13:
  v12 = *(_QWORD *)(a1 + 8);
  v13 = a2;
  v14 = *(unsigned int *)(a1 + 24);
  v15 = v14 - 1;
  while ( (_DWORD)v14 )
  {
    v16 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v17 = (__int64 *)(v12 + 16LL * v16);
    v18 = *v17;
    if ( v13 != *v17 )
    {
      v40 = 1;
      while ( v18 != -4096 )
      {
        v41 = v40 + 1;
        v16 = v15 & (v40 + v16);
        v17 = (__int64 *)(v12 + 16LL * v16);
        v18 = *v17;
        if ( v13 == *v17 )
          goto LABEL_15;
        v40 = v41;
      }
      break;
    }
LABEL_15:
    if ( (__int64 *)(v12 + 16 * v14) == v17 )
      break;
    v13 = v17[1];
  }
  if ( !(_BYTE)v6 )
    _libc_free((unsigned __int64)v61);
  if ( v57 != (char *)v59 )
    _libc_free((unsigned __int64)v57);
  return v13;
}
