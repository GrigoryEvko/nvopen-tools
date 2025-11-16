// Function: sub_CD02C0
// Address: 0xcd02c0
//
__int64 *__fastcall sub_CD02C0(__int64 *a1, _QWORD **a2, __int64 a3, __int64 a4, int a5)
{
  __int64 *v7; // rax
  __int64 *v8; // r14
  _QWORD *v9; // rsi
  __int64 v10; // rsi
  int v11; // r8d
  int v12; // r9d
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  __int64 *v15; // r15
  __int64 *v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 *v20; // r15
  __int64 *v21; // r14
  __int64 i; // rdx
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 *v26; // r14
  __int64 *v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rdi
  _BYTE *v30; // rbx
  _BYTE *v31; // r13
  _BYTE *v32; // rdi
  __int64 *v36; // [rsp+38h] [rbp-248h] BYREF
  __int64 v37[4]; // [rsp+40h] [rbp-240h] BYREF
  _BYTE v38[16]; // [rsp+60h] [rbp-220h] BYREF
  void (__fastcall *v39)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-210h]
  _QWORD v40[12]; // [rsp+80h] [rbp-200h] BYREF
  _QWORD v41[2]; // [rsp+E0h] [rbp-1A0h] BYREF
  _QWORD *v42; // [rsp+F0h] [rbp-190h]
  __int64 v43; // [rsp+F8h] [rbp-188h]
  _QWORD v44[3]; // [rsp+100h] [rbp-180h] BYREF
  int v45; // [rsp+118h] [rbp-168h]
  _QWORD *v46; // [rsp+120h] [rbp-160h]
  __int64 v47; // [rsp+128h] [rbp-158h]
  _QWORD v48[2]; // [rsp+130h] [rbp-150h] BYREF
  _QWORD *v49; // [rsp+140h] [rbp-140h]
  __int64 v50; // [rsp+148h] [rbp-138h]
  _QWORD v51[2]; // [rsp+150h] [rbp-130h] BYREF
  __int64 v52; // [rsp+160h] [rbp-120h]
  __int64 v53; // [rsp+168h] [rbp-118h]
  __int64 v54; // [rsp+170h] [rbp-110h]
  _BYTE *v55; // [rsp+178h] [rbp-108h]
  __int64 v56; // [rsp+180h] [rbp-100h]
  _BYTE v57[248]; // [rsp+188h] [rbp-F8h] BYREF

  if ( !a4 )
  {
    *a1 = 0;
    return a1;
  }
  v41[0] = 0;
  v42 = v44;
  v46 = v48;
  v49 = v51;
  v55 = v57;
  v41[1] = 0;
  v43 = 0;
  LOBYTE(v44[0]) = 0;
  v44[2] = 0;
  v45 = 0;
  v47 = 0;
  LOBYTE(v48[0]) = 0;
  v50 = 0;
  LOBYTE(v51[0]) = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v56 = 0x400000000LL;
  v7 = (__int64 *)sub_22077B0(8);
  v8 = v7;
  if ( v7 )
    sub_B6EEA0(v7);
  v9 = *a2;
  memset(v40, 0, 0x58u);
  sub_C7E010(v37, v9);
  v10 = (__int64)v41;
  sub_E46810(
    (unsigned int)&v36,
    (unsigned int)v41,
    (_DWORD)v8,
    (unsigned int)v38,
    v11,
    v12,
    v37[0],
    v37[1],
    v37[2],
    v37[3]);
  if ( LOBYTE(v40[10]) && (LOBYTE(v40[10]) = 0, v40[8]) )
  {
    v10 = (__int64)&v40[6];
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v40[8])(&v40[6], &v40[6], 3);
    if ( !LOBYTE(v40[5]) )
      goto LABEL_6;
  }
  else if ( !LOBYTE(v40[5]) )
  {
    goto LABEL_6;
  }
  LOBYTE(v40[5]) = 0;
  if ( !v40[3] )
  {
LABEL_6:
    if ( LOBYTE(v40[0]) )
      goto LABEL_51;
    goto LABEL_7;
  }
  v10 = (__int64)&v40[1];
  ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v40[3])(&v40[1], &v40[1], 3);
  if ( LOBYTE(v40[0]) )
  {
LABEL_51:
    LOBYTE(v40[0]) = 0;
    if ( v39 )
    {
      v10 = (__int64)v38;
      v39(v38, v38, 3);
    }
  }
LABEL_7:
  v15 = v36;
  v16 = 0;
  v36 = 0;
  if ( v15 )
  {
    v17 = sub_22077B0(152);
    v18 = v17;
    if ( v17 )
    {
      v10 = (__int64)v15;
      sub_CCFFF0(v17, v15, a3, a4, a5, 1, 1);
    }
    v13 = (_QWORD *)sub_22077B0(96);
    if ( v13 )
    {
      memset(v13, 0, 0x60u);
      v14 = 0;
      v13[11] = 1;
      v13[2] = v13 + 4;
      v13[3] = 0x400000000LL;
      v13[8] = v13 + 10;
    }
    v19 = *(_QWORD *)(v18 + 144);
    *(_QWORD *)(v18 + 144) = v13;
    if ( v19 )
    {
      v20 = *(__int64 **)(v19 + 16);
      v21 = &v20[*(unsigned int *)(v19 + 24)];
      if ( v20 != v21 )
      {
        for ( i = *(_QWORD *)(v19 + 16); ; i = *(_QWORD *)(v19 + 16) )
        {
          v23 = *v20;
          v24 = (unsigned int)(((__int64)v20 - i) >> 3) >> 7;
          v10 = 4096LL << v24;
          if ( v24 >= 0x1E )
            v10 = 0x40000000000LL;
          ++v20;
          sub_C7D6A0(v23, v10, 16);
          if ( v21 == v20 )
            break;
        }
      }
      v25 = *(__int64 **)(v19 + 64);
      v26 = &v25[2 * *(unsigned int *)(v19 + 72)];
      if ( v25 != v26 )
      {
        v27 = *(__int64 **)(v19 + 64);
        do
        {
          v10 = v27[1];
          v28 = *v27;
          v27 += 2;
          sub_C7D6A0(v28, v10, 16);
        }
        while ( v26 != v27 );
        v26 = *(__int64 **)(v19 + 64);
      }
      if ( v26 != (__int64 *)(v19 + 80) )
        _libc_free(v26, v10);
      v29 = *(_QWORD *)(v19 + 16);
      if ( v29 != v19 + 32 )
        _libc_free(v29, v10);
      v10 = 96;
      j_j___libc_free_0(v19, 96);
    }
    *a1 = v18;
    v16 = v36;
  }
  else
  {
    if ( v8 )
    {
      sub_B6E710(v8);
      v10 = 8;
      j_j___libc_free_0(v8, 8);
      v16 = v36;
    }
    *a1 = 0;
  }
  if ( v16 )
  {
    sub_BA9C10((_QWORD **)v16, v10, (__int64)v13, v14);
    v10 = 880;
    j_j___libc_free_0(v16, 880);
  }
  v30 = v55;
  v31 = &v55[48 * (unsigned int)v56];
  if ( v55 != v31 )
  {
    do
    {
      v31 -= 48;
      v32 = (_BYTE *)*((_QWORD *)v31 + 2);
      if ( v32 != v31 + 32 )
      {
        v10 = *((_QWORD *)v31 + 4) + 1LL;
        j_j___libc_free_0(v32, v10);
      }
    }
    while ( v30 != v31 );
    v31 = v55;
  }
  if ( v31 != v57 )
    _libc_free(v31, v10);
  if ( v52 )
    j_j___libc_free_0(v52, v54 - v52);
  if ( v49 != v51 )
    j_j___libc_free_0(v49, v51[0] + 1LL);
  if ( v46 != v48 )
    j_j___libc_free_0(v46, v48[0] + 1LL);
  if ( v42 != v44 )
    j_j___libc_free_0(v42, v44[0] + 1LL);
  return a1;
}
