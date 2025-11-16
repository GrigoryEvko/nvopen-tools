// Function: sub_ADCDB0
// Address: 0xadcdb0
//
void __fastcall sub_ADCDB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned int v7; // eax
  __int64 *v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r15
  _QWORD *v12; // rax
  __int64 *v13; // rdx
  char v14; // dl
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // r13
  __int64 *v18; // r14
  __int64 *v19; // r13
  __int64 *v20; // r14
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 *v24; // r13
  __int64 *i; // r15
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdi
  __int64 v33; // r14
  __int64 v34; // rsi
  __int64 v35; // rdx
  int v36; // r8d
  unsigned __int8 v37; // al
  __int64 *v38; // r13
  __int64 *v39; // r14
  __int64 v40; // rdi
  __int64 v41; // r14
  __int64 v42; // r13
  bool v43; // zf
  __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rax
  unsigned __int64 v50; // r8
  __int64 v51; // r14
  _QWORD *v52; // r13
  _QWORD *v53; // rax
  __int64 v54; // r9
  _QWORD *v55; // rcx
  __int64 v56; // rdi
  __int64 v57; // rax
  unsigned __int64 v58; // r8
  __int64 *v59; // r12
  __int64 v60; // r13
  __int64 *v61; // rax
  __int64 *v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // [rsp-10h] [rbp-210h]
  __int64 v66; // [rsp-8h] [rbp-208h]
  __int64 v67; // [rsp+0h] [rbp-200h]
  __int64 v68; // [rsp+8h] [rbp-1F8h]
  int v69; // [rsp+8h] [rbp-1F8h]
  int v70; // [rsp+8h] [rbp-1F8h]
  __int64 *v71; // [rsp+10h] [rbp-1F0h] BYREF
  __int64 v72; // [rsp+18h] [rbp-1E8h]
  _BYTE v73[128]; // [rsp+20h] [rbp-1E0h] BYREF
  _BYTE *v74; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-158h]
  _BYTE v76[128]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 *v77; // [rsp+130h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+138h] [rbp-C8h]
  __int64 v79; // [rsp+140h] [rbp-C0h] BYREF
  int v80; // [rsp+148h] [rbp-B8h]
  char v81; // [rsp+14Ch] [rbp-B4h]
  char v82; // [rsp+150h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  if ( !v6 )
    return;
  v7 = *(_DWORD *)(a1 + 64);
  if ( v7 )
  {
    v58 = v7;
    v59 = *(__int64 **)(a1 + 56);
    v78 = 0x1000000000LL;
    v60 = v7;
    v61 = &v79;
    v77 = &v79;
    if ( v58 > 0x10 )
    {
      v69 = v58;
      sub_C8D5F0(&v77, &v79, v58, 8);
      LODWORD(v58) = v69;
      v61 = &v77[(unsigned int)v78];
    }
    v62 = &v61[v60];
    do
    {
      if ( v61 )
        *v61 = *v59;
      ++v61;
      ++v59;
    }
    while ( v61 != v62 );
    v63 = *(_QWORD *)(a1 + 8);
    LODWORD(v78) = v78 + v58;
    v64 = sub_B9C770(v63, v77, (unsigned int)v78, 0, 1);
    a2 = 4;
    sub_BA6610(v6, 4, v64);
    if ( v77 != &v79 )
      _libc_free(v77, 4);
  }
  v9 = *(__int64 **)(a1 + 104);
  v77 = 0;
  v72 = 0x1000000000LL;
  v78 = (__int64)&v82;
  v10 = *(unsigned int *)(a1 + 112);
  v71 = (__int64 *)v73;
  v11 = &v9[v10];
  v81 = 1;
  v79 = 16;
  v80 = 0;
  if ( v9 == v11 )
    goto LABEL_16;
  a2 = *v9;
LABEL_5:
  v12 = (_QWORD *)v78;
  a4 = HIDWORD(v79);
  v13 = (__int64 *)(v78 + 8LL * HIDWORD(v79));
  if ( (__int64 *)v78 == v13 )
  {
LABEL_35:
    if ( HIDWORD(v79) >= (unsigned int)v79 )
      goto LABEL_11;
    ++HIDWORD(v79);
    *v13 = a2;
    v77 = (__int64 *)((char *)v77 + 1);
LABEL_12:
    v15 = (unsigned int)v72;
    a4 = HIDWORD(v72);
    a5 = *v9;
    v16 = (unsigned int)v72 + 1LL;
    if ( v16 > HIDWORD(v72) )
    {
      a2 = (__int64)v73;
      v68 = *v9;
      sub_C8D5F0(&v71, v73, v16, 8);
      v15 = (unsigned int)v72;
      a5 = v68;
    }
    ++v9;
    v71[v15] = a5;
    LODWORD(v72) = v72 + 1;
    if ( v11 != v9 )
      goto LABEL_10;
  }
  else
  {
    while ( a2 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_35;
    }
    while ( v11 != ++v9 )
    {
LABEL_10:
      a2 = *v9;
      if ( v81 )
        goto LABEL_5;
LABEL_11:
      sub_C8CC70(&v77, a2);
      if ( v14 )
        goto LABEL_12;
    }
  }
  if ( (_DWORD)v72 )
  {
    v46 = *(_QWORD *)(a1 + 16);
    v47 = sub_B9C770(*(_QWORD *)(a1 + 8), v71, (unsigned int)v72, 0, 1);
    a2 = 5;
    sub_BA6610(v46, 5, v47);
    v17 = *(__int64 **)(a1 + 152);
    v18 = &v17[*(unsigned int *)(a1 + 160)];
    if ( v18 == v17 )
      goto LABEL_18;
  }
  else
  {
LABEL_16:
    v17 = *(__int64 **)(a1 + 152);
    v18 = &v17[*(unsigned int *)(a1 + 160)];
    if ( v18 == v17 )
      goto LABEL_22;
  }
  do
  {
    a2 = *v17++;
    sub_ADC590(a1, a2);
  }
  while ( v18 != v17 );
LABEL_18:
  v19 = v71;
  v20 = &v71[(unsigned int)v72];
  if ( v20 != v71 )
  {
    do
    {
      a2 = *v19;
      if ( *(_BYTE *)*v19 == 18 )
        sub_ADC590(a1, a2);
      ++v19;
    }
    while ( v20 != v19 );
  }
LABEL_22:
  v21 = *(_DWORD *)(a1 + 208);
  if ( v21 )
  {
    v48 = *(_QWORD *)(a1 + 16);
    v49 = sub_B9C770(*(_QWORD *)(a1 + 8), *(_QWORD *)(a1 + 200), v21, 0, 1);
    a2 = 6;
    sub_BA6610(v48, 6, v49);
    v22 = *(_DWORD *)(a1 + 256);
    if ( !v22 )
      goto LABEL_24;
LABEL_57:
    v50 = v22;
    v51 = *(_QWORD *)(a1 + 16);
    v75 = 0x1000000000LL;
    v52 = *(_QWORD **)(a1 + 248);
    v53 = v76;
    v54 = 8 * v50;
    v74 = v76;
    if ( v50 > 0x10 )
    {
      v67 = 8 * v50;
      v70 = v50;
      sub_C8D5F0(&v74, v76, v50, 8);
      v54 = v67;
      LODWORD(v50) = v70;
      v53 = &v74[8 * (unsigned int)v75];
    }
    v55 = (_QWORD *)((char *)v53 + v54);
    do
    {
      if ( v53 )
        *v53 = *v52;
      ++v53;
      ++v52;
    }
    while ( v55 != v53 );
    v56 = *(_QWORD *)(a1 + 8);
    LODWORD(v75) = v75 + v50;
    v57 = sub_B9C770(v56, v74, (unsigned int)v75, 0, 1);
    a2 = 7;
    sub_BA6610(v51, 7, v57);
    if ( v74 != v76 )
      _libc_free(v74, 7);
    goto LABEL_24;
  }
  v22 = *(_DWORD *)(a1 + 256);
  if ( v22 )
    goto LABEL_57;
LABEL_24:
  v23 = *(unsigned int *)(a1 + 336);
  v24 = *(__int64 **)(a1 + 328);
  for ( i = &v24[7 * v23]; i != v24; v24 += 7 )
  {
    v33 = *v24;
    v34 = v24[5];
    v35 = *((unsigned int *)v24 + 12);
    if ( *v24 )
    {
      v36 = sub_ADCD90(a1, v34, v35);
      v37 = *(_BYTE *)(v33 - 16);
      if ( (v37 & 2) != 0 )
        v26 = *(_QWORD **)(v33 - 32);
      else
        v26 = (_QWORD *)(v33 - 16 - 8LL * ((v37 >> 2) & 0xF));
      a2 = 3;
      v27 = sub_B10A70(*(_QWORD *)(a1 + 8), 3, *(_DWORD *)(v33 + 4), *v26, v36, 0, 1);
      v30 = v65;
      v31 = v66;
      if ( v33 == v27 )
      {
        sub_BA6670(v33, 3, v65);
      }
      else
      {
        v32 = *(_QWORD *)(v33 + 8);
        if ( (v32 & 4) != 0 )
        {
          a2 = v27;
          sub_BA6110(v32 & 0xFFFFFFFFFFFFFFF8LL, v27);
        }
        sub_BA65D0(v33, a2, v30, v31, v28, v29);
      }
    }
    else
    {
      v44 = *(_QWORD *)(a1 + 16);
      v45 = sub_B9C770(*(_QWORD *)(a1 + 8), v34, v35, 0, 1);
      a2 = 8;
      sub_BA6610(v44, 8, v45);
    }
  }
  v38 = *(__int64 **)(a1 + 344);
  v39 = &v38[*(unsigned int *)(a1 + 352)];
  if ( v39 != v38 )
  {
    do
    {
      v40 = *v38;
      if ( *v38 && ((*(_BYTE *)(v40 + 1) & 0x7F) == 2 || *(_DWORD *)(v40 - 8)) )
        sub_B931A0(v40, a2, v23, a4, a5, a6);
      ++v38;
    }
    while ( v39 != v38 );
    v41 = *(_QWORD *)(a1 + 344);
    v42 = v41 + 8LL * *(unsigned int *)(a1 + 352);
    while ( v41 != v42 )
    {
      while ( 1 )
      {
        a2 = *(_QWORD *)(v42 - 8);
        v42 -= 8;
        if ( !a2 )
          break;
        sub_B91220(v42);
        if ( v41 == v42 )
          goto LABEL_47;
      }
    }
  }
LABEL_47:
  v43 = v81 == 0;
  *(_BYTE *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  if ( v43 )
    _libc_free(v78, a2);
  if ( v71 != (__int64 *)v73 )
    _libc_free(v71, a2);
}
