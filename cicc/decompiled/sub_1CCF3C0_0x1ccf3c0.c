// Function: sub_1CCF3C0
// Address: 0x1ccf3c0
//
__int64 *__fastcall sub_1CCF3C0(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        int *a4,
        char a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        __int64 a15)
{
  char v18; // r8
  __int64 *result; // rax
  int v20; // r15d
  unsigned int v21; // r8d
  __int64 v22; // rax
  unsigned __int8 v23; // r14
  unsigned int v24; // r15d
  __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  unsigned __int64 v29; // r8
  __int64 v30; // r13
  __int64 v31; // r12
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  unsigned int v34; // r8d
  __int64 *v35; // rdx
  unsigned int v36; // eax
  __int64 *v37; // rdx
  __int64 v38; // rax
  __int64 *v39; // rcx
  unsigned int v40; // eax
  __int64 *v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // rax
  __int64 *v46; // rax
  __int64 *v47; // [rsp+10h] [rbp-80h]
  unsigned int v48; // [rsp+18h] [rbp-78h]
  __int64 *v49; // [rsp+18h] [rbp-78h]
  _QWORD v50[4]; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v51; // [rsp+40h] [rbp-50h] BYREF
  __int64 v52; // [rsp+48h] [rbp-48h]
  __int64 v53; // [rsp+50h] [rbp-40h]

  v18 = sub_1CCCEA0(a1, a2, a3, a6, a7, a8, a9, a10, a11, a12, a13);
  result = 0;
  if ( v18 )
    return result;
  v51 = 0;
  v53 = 0x1000000000LL;
  v20 = *a4;
  v52 = 0;
  v21 = sub_16D19C0((__int64)&v51, "__CUDA_ARCH", 0xBu);
  v22 = *(_QWORD *)(v51 + 8LL * v21);
  if ( v22 )
  {
    if ( v22 != -8 )
      goto LABEL_4;
    LODWORD(v53) = v53 - 1;
  }
  v47 = (__int64 *)(v51 + 8LL * v21);
  v48 = v21;
  v33 = malloc(0x1Cu);
  v34 = v48;
  v35 = v47;
  if ( !v33 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v33 = 0;
    v35 = v47;
    v34 = v48;
  }
  strcpy((char *)(v33 + 16), "__CUDA_ARCH");
  *(_QWORD *)v33 = 11;
  *(_DWORD *)(v33 + 8) = 0;
  *v35 = v33;
  ++HIDWORD(v52);
  v36 = sub_16D1CD0((__int64)&v51, v34);
  v37 = (__int64 *)(v51 + 8LL * v36);
  v22 = *v37;
  if ( *v37 )
    goto LABEL_22;
  do
  {
    do
    {
      v22 = v37[1];
      ++v37;
    }
    while ( !v22 );
LABEL_22:
    ;
  }
  while ( v22 == -8 );
LABEL_4:
  *(_DWORD *)(v22 + 8) = v20;
  v23 = *((_BYTE *)a4 + 4);
  v24 = sub_16D19C0((__int64)&v51, "__CUDA_FTZ", 0xAu);
  v25 = *(_QWORD *)(v51 + 8LL * v24);
  if ( !v25 )
  {
LABEL_25:
    v49 = (__int64 *)(v51 + 8LL * v24);
    v38 = malloc(0x1Bu);
    v39 = v49;
    if ( !v38 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v38 = 0;
      v39 = v49;
    }
    strcpy((char *)(v38 + 16), "__CUDA_FTZ");
    *(_QWORD *)v38 = 10;
    *(_DWORD *)(v38 + 8) = 0;
    *v39 = v38;
    ++HIDWORD(v52);
    v40 = sub_16D1CD0((__int64)&v51, v24);
    v41 = (__int64 *)(v51 + 8LL * v40);
    v25 = *v41;
    if ( *v41 == -8 || !v25 )
    {
      v42 = v41 + 1;
      do
      {
        do
          v25 = *v42++;
        while ( !v25 );
      }
      while ( v25 == -8 );
    }
    goto LABEL_6;
  }
  if ( v25 == -8 )
  {
    LODWORD(v53) = v53 - 1;
    goto LABEL_25;
  }
LABEL_6:
  *(_DWORD *)(v25 + 8) = v23;
  sub_1611730(v50, a15);
  v26 = sub_1CB9110((__int64 *)&v51);
  sub_1619140((__int64)v50, v26, 0);
  v27 = (__int64 *)sub_1CC63C0();
  sub_1619140((__int64)v50, v27, 0);
  v28 = (__int64 *)sub_1857160();
  sub_1619140((__int64)v50, v28, 0);
  if ( a5 )
  {
    v43 = (__int64 *)sub_17060B0(1, 0);
    sub_1619140((__int64)v50, v43, 0);
    v44 = (__int64 *)sub_1869BD0();
    sub_1619140((__int64)v50, v44, 0);
    v45 = (__int64 *)sub_1B26330();
    sub_1619140((__int64)v50, v45, 0);
    v46 = (__int64 *)sub_1A223D0();
    sub_1619140((__int64)v50, v46, 0);
  }
  sub_1619BD0((__int64)v50, (char *)a1);
  sub_160FE50(v50);
  if ( HIDWORD(v52) )
  {
    v29 = v51;
    if ( (_DWORD)v52 )
    {
      v30 = 8LL * (unsigned int)v52;
      v31 = 0;
      do
      {
        v32 = *(_QWORD *)(v29 + v31);
        if ( v32 )
        {
          if ( v32 != -8 )
          {
            _libc_free(v32);
            v29 = v51;
          }
        }
        v31 += 8;
      }
      while ( v31 != v30 );
    }
    _libc_free(v29);
    return a1;
  }
  else
  {
    _libc_free(v51);
    return a1;
  }
}
