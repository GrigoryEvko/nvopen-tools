// Function: sub_38BEE30
// Address: 0x38bee30
//
__int64 __fastcall sub_38BEE30(__int64 a1, void *a2, unsigned __int64 a3, char a4, char a5, int a6)
{
  int v9; // eax
  unsigned int v10; // edx
  int v11; // r9d
  char v12; // r8
  __int64 *v13; // r10
  __int64 v14; // r15
  _QWORD *v15; // r10
  int v16; // r13d
  size_t v17; // r12
  unsigned int v18; // eax
  void *v19; // r10
  unsigned int v20; // r11d
  __int64 *v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // r8d
  __int64 *v27; // r10
  __int64 v28; // rcx
  void *v29; // rdi
  __int64 *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rax
  unsigned int v33; // r11d
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v38; // rax
  void *v39; // rax
  __int64 v40; // rax
  size_t v41; // rdx
  char *v42; // rdi
  char *v43; // rax
  char *i; // rdx
  size_t v45; // rdx
  __int64 v46; // [rsp+0h] [rbp-1A0h]
  __int64 *v47; // [rsp+8h] [rbp-198h]
  __int64 v48; // [rsp+8h] [rbp-198h]
  __int64 *v49; // [rsp+8h] [rbp-198h]
  void *src; // [rsp+10h] [rbp-190h]
  _QWORD *v51; // [rsp+18h] [rbp-188h]
  unsigned int v52; // [rsp+18h] [rbp-188h]
  unsigned int v53; // [rsp+18h] [rbp-188h]
  __int64 *v54; // [rsp+18h] [rbp-188h]
  unsigned int v55; // [rsp+18h] [rbp-188h]
  char v56; // [rsp+24h] [rbp-17Ch]
  __int64 v57; // [rsp+30h] [rbp-170h]
  unsigned int v58; // [rsp+30h] [rbp-170h]
  void *v59; // [rsp+40h] [rbp-160h] BYREF
  size_t n; // [rsp+48h] [rbp-158h]
  _BYTE v61[128]; // [rsp+50h] [rbp-150h] BYREF
  _QWORD *v62; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v63; // [rsp+D8h] [rbp-C8h]
  _QWORD v64[2]; // [rsp+E0h] [rbp-C0h] BYREF
  int v65; // [rsp+F0h] [rbp-B0h]
  void **v66; // [rsp+F8h] [rbp-A8h]
  char v67; // [rsp+160h] [rbp-40h]

  if ( a5 )
  {
    v56 = 1;
    if ( !*(_BYTE *)(a1 + 1163) )
      return sub_38BD490(a1, 0, 1);
  }
  else if ( *(_BYTE *)(a1 + 1162) )
  {
    v56 = 0;
    v40 = *(_QWORD *)(a1 + 16);
    v41 = *(_QWORD *)(v40 + 88);
    if ( v41 <= a3 )
    {
      v56 = 1;
      if ( v41 )
        v56 = memcmp(a2, *(const void **)(v40 + 80), v41) == 0;
    }
  }
  else
  {
    v56 = 0;
  }
  v59 = v61;
  n = 0x8000000000LL;
  if ( a3 > 0x80 )
  {
    sub_16CD150((__int64)&v59, v61, a3, 1, a5, a6);
    v42 = (char *)v59 + (unsigned int)n;
  }
  else
  {
    v9 = 0;
    if ( !a3 )
      goto LABEL_5;
    v42 = v61;
  }
  memcpy(v42, a2, a3);
  v9 = a3 + n;
LABEL_5:
  LODWORD(n) = v9;
  v10 = sub_16D19C0(a1 + 664, (unsigned __int8 *)a2, a3);
  v12 = v10;
  v13 = (__int64 *)(*(_QWORD *)(a1 + 664) + 8LL * v10);
  v14 = *v13;
  if ( *v13 )
  {
    if ( v14 != -8 )
      goto LABEL_7;
    --*(_DWORD *)(a1 + 680);
  }
  v47 = v13;
  v52 = v10;
  v25 = malloc(a3 + 17);
  v26 = v52;
  v27 = v47;
  v28 = v25;
  if ( !v25 )
  {
    if ( a3 == -17 )
    {
      v38 = malloc(1u);
      v26 = v52;
      v27 = v47;
      v28 = 0;
      if ( v38 )
      {
        v29 = (void *)(v38 + 16);
        v28 = v38;
        goto LABEL_44;
      }
    }
    v46 = v28;
    v49 = v27;
    v55 = v26;
    sub_16BD1C0("Allocation failed", 1u);
    v26 = v55;
    v27 = v49;
    v28 = v46;
  }
  v29 = (void *)(v28 + 16);
  if ( a3 + 1 > 1 )
  {
LABEL_44:
    v48 = v28;
    v54 = v27;
    v58 = v26;
    v39 = memcpy(v29, a2, a3);
    v28 = v48;
    v27 = v54;
    v26 = v58;
    v29 = v39;
  }
  *((_BYTE *)v29 + a3) = 0;
  *(_QWORD *)v28 = a3;
  *(_DWORD *)(v28 + 8) = 0;
  *v27 = v28;
  ++*(_DWORD *)(a1 + 676);
  v30 = (__int64 *)(*(_QWORD *)(a1 + 664) + 8LL * (unsigned int)sub_16D1CD0(a1 + 664, v26));
  v14 = *v30;
  if ( !*v30 || v14 == -8 )
  {
    v31 = v30 + 1;
    do
    {
      do
        v14 = *v31++;
      while ( !v14 );
    }
    while ( v14 == -8 );
  }
LABEL_7:
  v57 = a1 + 632;
  if ( a4 )
    goto LABEL_15;
  while ( 1 )
  {
    v15 = v64;
    v16 = n;
    v63 = 0x8000000000LL;
    v62 = v64;
    if ( !(_DWORD)n )
    {
      v17 = 0;
      goto LABEL_10;
    }
    v17 = (unsigned int)n;
    if ( (unsigned int)n > 0x80 )
    {
      sub_16CD150((__int64)&v62, v64, (unsigned int)n, 1, v12, v11);
      v45 = (unsigned int)n;
      v15 = v62;
      if ( !(_DWORD)n )
        goto LABEL_63;
    }
    else
    {
      v15 = v64;
      v45 = (unsigned int)n;
    }
    memcpy(v15, v59, v45);
    v15 = v62;
LABEL_63:
    LODWORD(v63) = v16;
LABEL_10:
    v51 = v15;
    v67 = 1;
    v18 = sub_16D19C0(v57, (unsigned __int8 *)v15, v17);
    v19 = v51;
    v20 = v18;
    v21 = (__int64 *)(*(_QWORD *)(a1 + 632) + 8LL * v18);
    v22 = *v21;
    if ( !*v21 )
      goto LABEL_28;
    if ( v22 == -8 )
      break;
    if ( v62 != v64 )
    {
      _libc_free((unsigned __int64)v62);
      v22 = *v21;
    }
    if ( !*(_BYTE *)(v22 + 8) )
      goto LABEL_36;
LABEL_15:
    v23 = (unsigned int)n;
    if ( a3 < (unsigned int)n )
      goto LABEL_16;
    if ( a3 > (unsigned int)n )
    {
      if ( a3 > HIDWORD(n) )
      {
        sub_16CD150((__int64)&v59, v61, a3, 1, v12, v11);
        v23 = (unsigned int)n;
      }
      v43 = (char *)v59 + v23;
      for ( i = (char *)v59 + a3; i != v43; ++v43 )
      {
        if ( v43 )
          *v43 = 0;
      }
LABEL_16:
      LODWORD(n) = a3;
    }
    v65 = 1;
    v64[1] = 0;
    v62 = &unk_49EFC48;
    v64[0] = 0;
    v66 = &v59;
    v63 = 0;
    sub_16E7A40((__int64)&v62, 0, 0, 0);
    v24 = *(unsigned int *)(v14 + 8);
    *(_DWORD *)(v14 + 8) = v24 + 1;
    sub_16E7A90((__int64)&v62, v24);
    v62 = &unk_49EFD28;
    sub_16E7960((__int64)&v62);
  }
  --*(_DWORD *)(a1 + 648);
LABEL_28:
  v53 = v20;
  src = v19;
  v32 = sub_145CBF0(*(__int64 **)(a1 + 656), v17 + 17, 8);
  v33 = v53;
  v34 = v32;
  if ( v17 )
  {
    memcpy((void *)(v32 + 16), src, v17);
    v33 = v53;
  }
  *(_BYTE *)(v34 + v17 + 16) = 0;
  *(_QWORD *)v34 = v17;
  *(_BYTE *)(v34 + 8) = 1;
  *v21 = v34;
  ++*(_DWORD *)(a1 + 644);
  v21 = (__int64 *)(*(_QWORD *)(a1 + 632) + 8LL * (unsigned int)sub_16D1CD0(v57, v33));
  v35 = *v21;
  if ( *v21 != -8 )
    goto LABEL_32;
  do
  {
    do
    {
      v35 = v21[1];
      ++v21;
    }
    while ( v35 == -8 );
LABEL_32:
    ;
  }
  while ( !v35 );
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  v22 = *v21;
LABEL_36:
  *(_BYTE *)(v22 + 8) = 1;
  v36 = sub_38BD490(a1, *v21, v56);
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  return v36;
}
