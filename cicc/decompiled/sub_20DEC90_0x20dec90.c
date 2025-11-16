// Function: sub_20DEC90
// Address: 0x20dec90
//
__int64 __fastcall sub_20DEC90(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 v7; // r10
  void *v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 (*v16)(); // rax
  __int64 v17; // r15
  _QWORD *v18; // rbx
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  __int64 v21; // rax
  int v22; // r12d
  int v23; // r14d
  __int64 v24; // r10
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // r15
  __int64 v27; // rax
  int v28; // r9d
  __int64 v29; // r10
  void *v30; // r14
  _QWORD *i; // r15
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 j; // rdi
  char v35; // r8
  __int64 v36; // rsi
  unsigned int v37; // ecx
  _QWORD *v38; // rax
  _QWORD *v40; // r15
  int v41; // r14d
  _QWORD *v42; // rsi
  __int64 v43; // r15
  int v44; // r14d
  __int64 v45; // rsi
  char v46; // al
  __int64 v47; // rax
  __int64 v48; // [rsp+0h] [rbp-110h]
  int v49; // [rsp+0h] [rbp-110h]
  int v50; // [rsp+0h] [rbp-110h]
  int v51; // [rsp+0h] [rbp-110h]
  int v52; // [rsp+8h] [rbp-108h]
  __int64 v53; // [rsp+8h] [rbp-108h]
  __int64 v54; // [rsp+8h] [rbp-108h]
  __int64 v55; // [rsp+8h] [rbp-108h]
  unsigned __int8 v57; // [rsp+10h] [rbp-100h]
  __int64 v58; // [rsp+10h] [rbp-100h]
  __int64 v60; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v61; // [rsp+28h] [rbp-E8h] BYREF
  __int64 *v62; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+38h] [rbp-D8h]
  __int64 v64; // [rsp+40h] [rbp-D0h] BYREF
  int v65; // [rsp+48h] [rbp-C8h]

  v7 = a4;
  ++*(_QWORD *)(a1 + 24);
  v12 = *(void **)(a1 + 40);
  if ( v12 == *(void **)(a1 + 32) )
    goto LABEL_6;
  v13 = 4 * (*(_DWORD *)(a1 + 52) - *(_DWORD *)(a1 + 56));
  v14 = *(unsigned int *)(a1 + 48);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    memset(v12, -1, 8 * v14);
    v7 = a4;
LABEL_6:
    *(_QWORD *)(a1 + 52) = 0;
    goto LABEL_7;
  }
  sub_16CC920(a1 + 24);
  v7 = a4;
LABEL_7:
  v15 = a2[5];
  *(_BYTE *)(a1 + 136) = a7;
  *(_QWORD *)(a1 + 144) = a3;
  *(_QWORD *)(a1 + 160) = v7;
  *(_QWORD *)(a1 + 168) = a5;
  *(_QWORD *)(a1 + 176) = a6;
  *(_QWORD *)(a1 + 152) = v15;
  if ( (**(_BYTE **)(*(_QWORD *)v15 + 352LL) & 4) != 0
    && (v16 = *(__int64 (**)())(*(_QWORD *)v7 + 328LL), v16 != sub_1F49C90)
    && (v58 = v15, v46 = ((__int64 (__fastcall *)(__int64, _QWORD *))v16)(v7, a2), v15 = v58, v46) )
  {
    *(_BYTE *)(a1 + 139) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 139) = 0;
    **(_QWORD **)(*(_QWORD *)v15 + 352LL) &= ~4uLL;
  }
  v57 = 0;
  v17 = a2[41];
  v18 = a2 + 40;
  if ( (_QWORD *)v17 != a2 + 40 )
  {
    do
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(a1 + 144);
        v62 = &v64;
        v63 = 0x400000000LL;
        v60 = 0;
        v61 = 0;
        v20 = *(__int64 (**)())(*(_QWORD *)v19 + 264LL);
        if ( v20 != sub_1D820E0 )
        {
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, __int64 **, __int64))v20)(
                  v19,
                  v17,
                  &v60,
                  &v61,
                  &v62,
                  1) )
            v57 |= sub_1DD9390(v17, v60, v61, (_DWORD)v63 != 0);
          if ( v62 != &v64 )
            break;
        }
        v17 = *(_QWORD *)(v17 + 8);
        if ( v18 == (_QWORD *)v17 )
          goto LABEL_18;
      }
      _libc_free((unsigned __int64)v62);
      v17 = *(_QWORD *)(v17 + 8);
    }
    while ( v18 != (_QWORD *)v17 );
  }
LABEL_18:
  sub_20C9140((__int64)&v62, (__int64)a2);
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  v21 = v63;
  ++*(_QWORD *)(a1 + 80);
  v62 = (__int64 *)((char *)v62 + 1);
  *(_QWORD *)(a1 + 88) = v21;
  v63 = 0;
  *(_QWORD *)(a1 + 96) = v64;
  v64 = 0;
  *(_DWORD *)(a1 + 104) = v65;
  v65 = 0;
  j___libc_free_0(0);
  while ( 1 )
  {
    v23 = sub_20DE010(a1, (__int64)a2);
    LOBYTE(v22) = v23 | *(_BYTE *)(a1 + 136) ^ 1;
    if ( !(_BYTE)v22 )
      break;
    v22 = sub_20DA520(a1, a2) | v23;
    if ( !byte_4FCF2E0 )
      goto LABEL_20;
LABEL_51:
    v40 = (_QWORD *)a2[41];
    if ( v18 == v40 )
    {
LABEL_20:
      if ( !*(_BYTE *)(a1 + 138) )
        goto LABEL_21;
LABEL_55:
      v43 = a2[41];
      if ( v18 == (_QWORD *)v43 )
        goto LABEL_21;
LABEL_56:
      v44 = 0;
      do
      {
        v45 = v43;
        v43 = *(_QWORD *)(v43 + 8);
        v44 |= sub_20DA7F0(a1, v45);
      }
      while ( v18 != (_QWORD *)v43 );
      LOBYTE(v22) = v44 | v22;
      goto LABEL_21;
    }
    v41 = 0;
    do
    {
      v42 = v40;
      v40 = (_QWORD *)v40[1];
      v41 |= sub_20D8130(a1, v42);
    }
    while ( v18 != v40 );
    LOBYTE(v22) = v41 | v22;
    if ( *(_BYTE *)(a1 + 138) )
      goto LABEL_55;
LABEL_21:
    if ( !(_BYTE)v22 )
      goto LABEL_26;
    v57 = v22;
  }
  if ( byte_4FCF2E0 )
    goto LABEL_51;
  if ( *(_BYTE *)(a1 + 138) )
  {
    v43 = a2[41];
    if ( v18 != (_QWORD *)v43 )
      goto LABEL_56;
  }
LABEL_26:
  v24 = a2[9];
  if ( v24 )
  {
    v48 = a2[9];
    v52 = -1431655765 * ((__int64)(*(_QWORD *)(v24 + 16) - *(_QWORD *)(v24 + 8)) >> 3);
    v25 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(v24 + 16) - *(_QWORD *)(v24 + 8)) >> 3);
    v26 = 8LL * ((unsigned int)(v25 + 63) >> 6);
    v27 = malloc(v26);
    v28 = v52;
    v29 = v48;
    v30 = (void *)v27;
    if ( !v27 )
    {
      if ( v26 || (v51 = v52, v55 = v29, v47 = malloc(1u), v29 = v55, v28 = v51, !v47) )
      {
        v50 = v28;
        v54 = v29;
        sub_16BD1C0("Allocation failed", 1u);
        v29 = v54;
        v28 = v50;
      }
      else
      {
        v30 = (void *)v47;
      }
    }
    if ( (unsigned int)(v25 + 63) >> 6 )
    {
      v49 = v28;
      v53 = v29;
      memset(v30, 0, v26);
      v28 = v49;
      v29 = v53;
    }
    for ( i = (_QWORD *)a2[41]; v18 != i; i = (_QWORD *)i[1] )
    {
      v32 = i[4];
      if ( (_QWORD *)v32 != i + 3 )
      {
        do
        {
          v33 = *(_QWORD *)(v32 + 32);
          for ( j = v33 + 40LL * *(unsigned int *)(v32 + 40); j != v33; v33 += 40 )
          {
            if ( *(_BYTE *)v33 == 8 )
              *((_QWORD *)v30 + (*(_DWORD *)(v33 + 24) >> 6)) |= 1LL << *(_DWORD *)(v33 + 24);
          }
          if ( (*(_BYTE *)v32 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v32 + 46) & 8) != 0 )
              v32 = *(_QWORD *)(v32 + 8);
          }
          v32 = *(_QWORD *)(v32 + 8);
        }
        while ( i + 3 != (_QWORD *)v32 );
      }
    }
    if ( (_DWORD)v25 )
    {
      v35 = v57;
      v36 = 0;
      v37 = 0;
      do
      {
        if ( (*((_QWORD *)v30 + (v37 >> 6)) & (1LL << v37)) == 0 )
        {
          v35 = 1;
          v38 = (_QWORD *)(v36 + *(_QWORD *)(v29 + 8));
          if ( *v38 != v38[1] )
            v38[1] = *v38;
        }
        ++v37;
        v36 += 24;
      }
      while ( v28 != v37 );
      v57 = v35;
    }
    _libc_free((unsigned __int64)v30);
  }
  return v57;
}
