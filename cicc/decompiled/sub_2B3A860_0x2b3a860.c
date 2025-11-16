// Function: sub_2B3A860
// Address: 0x2b3a860
//
void __fastcall sub_2B3A860(__int64 a1, int a2, _DWORD *a3, __int64 a4, _DWORD *a5, __int64 a6)
{
  int v6; // r10d
  __int64 v11; // r8
  unsigned __int64 v12; // r14
  char *v13; // rdx
  char *i; // rax
  unsigned int v15; // r11d
  unsigned __int64 v16; // rdx
  _BYTE *v17; // rcx
  char *v18; // rax
  __int64 v19; // rdx
  char *j; // rcx
  _BYTE *v21; // rcx
  char *v22; // r11
  __int64 v23; // rax
  unsigned __int64 v24; // r11
  __int64 v25; // rdx
  _DWORD *v26; // rcx
  char *v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rdx
  int v31; // esi
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // r10d
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned int v37; // eax
  __int64 v38; // rdx
  size_t v39; // r9
  __int64 v40; // [rsp+8h] [rbp-F8h]
  int v41; // [rsp+10h] [rbp-F0h]
  __int64 v42; // [rsp+10h] [rbp-F0h]
  unsigned int v43; // [rsp+1Ch] [rbp-E4h]
  unsigned __int64 v44; // [rsp+20h] [rbp-E0h]
  __int64 v45; // [rsp+20h] [rbp-E0h]
  unsigned int v46; // [rsp+20h] [rbp-E0h]
  int v47; // [rsp+34h] [rbp-CCh] BYREF
  unsigned int **v48; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD v49[2]; // [rsp+40h] [rbp-C0h] BYREF
  void *dest; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v51; // [rsp+58h] [rbp-A8h]
  _BYTE v52[48]; // [rsp+60h] [rbp-A0h] BYREF
  void *v53; // [rsp+90h] [rbp-70h] BYREF
  __int64 v54; // [rsp+98h] [rbp-68h]
  _BYTE v55[96]; // [rsp+A0h] [rbp-60h] BYREF

  v6 = a2;
  v11 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v11 )
  {
    v12 = (unsigned int)v11;
    dest = v52;
    v51 = 0xC00000000LL;
    if ( (unsigned int)v11 <= 0xCuLL )
      goto LABEL_4;
LABEL_47:
    v42 = a6;
    v46 = v11;
    sub_C8D5F0((__int64)&dest, v52, v12, 4u, v11, a6);
    a6 = v42;
    v6 = a2;
    v11 = v46;
    goto LABEL_4;
  }
  v12 = *(unsigned int *)(a1 + 8);
  dest = v52;
  v11 = v12;
  v51 = 0xC00000000LL;
  if ( !v12 )
  {
    v54 = 0xC00000000LL;
    v53 = v55;
    goto LABEL_45;
  }
  if ( v12 > 0xC )
    goto LABEL_47;
LABEL_4:
  v13 = (char *)dest + 4 * v12;
  for ( i = (char *)dest + 4 * (unsigned int)v51; v13 != i; i += 4 )
  {
    if ( i )
      *(_DWORD *)i = 0;
  }
  v15 = *(_DWORD *)(a1 + 120);
  LODWORD(v51) = v11;
  if ( v15 )
  {
    v16 = v15;
    v54 = 0xC00000000LL;
    v17 = v55;
    v18 = v55;
    v53 = v55;
    if ( v15 <= 0xCuLL )
      goto LABEL_15;
    goto LABEL_10;
  }
  v16 = *(unsigned int *)(a1 + 8);
  v53 = v55;
  v54 = 0xC00000000LL;
  v15 = v16;
  if ( v16 )
  {
    v18 = v55;
    v17 = v55;
    if ( v16 <= 0xC )
    {
LABEL_15:
      v19 = 4 * v16;
      for ( j = &v17[v19]; j != v18; v18 += 4 )
      {
        if ( v18 )
          *(_DWORD *)v18 = 0;
      }
      v21 = dest;
      LODWORD(v54) = v15;
      v22 = (char *)dest + 4 * (unsigned int)v51;
      if ( dest == v22 )
        goto LABEL_23;
      goto LABEL_20;
    }
LABEL_10:
    v40 = a6;
    v41 = v6;
    v43 = v15;
    v44 = v16;
    sub_C8D5F0((__int64)&v53, v55, v16, 4u, v11, a6);
    v17 = v53;
    a6 = v40;
    v6 = v41;
    v15 = v43;
    v16 = v44;
    v18 = (char *)v53 + 4 * (unsigned int)v54;
    goto LABEL_15;
  }
LABEL_45:
  v21 = dest;
  v22 = (char *)dest + 4 * (unsigned int)v51;
  if ( dest == v22 )
    goto LABEL_27;
LABEL_20:
  v23 = 0;
  v24 = (unsigned __int64)(v22 - 4 - v21) >> 2;
  do
  {
    v25 = v23;
    *(_DWORD *)&v21[4 * v23] = v23;
    ++v23;
  }
  while ( v24 != v25 );
  v19 = 4LL * (unsigned int)v54;
LABEL_23:
  v26 = v53;
  v27 = (char *)v53 + v19;
  if ( v53 != v27 )
  {
    v28 = 0;
    v29 = (unsigned __int64)(v27 - (_BYTE *)v53 - 4) >> 2;
    do
    {
      v30 = v28;
      v26[v28] = v28;
      ++v28;
    }
    while ( v29 != v30 );
  }
  v22 = (char *)dest;
LABEL_27:
  if ( v6 )
  {
    v31 = *(_DWORD *)(*(_QWORD *)(a1 + 208) + 8LL * *(unsigned int *)(a1 + 216) - 4);
    if ( (_DWORD)a4 )
    {
      v32 = 0;
      do
      {
        v33 = (unsigned int)(v31 + v32);
        *(_DWORD *)&v22[4 * v33] = v31 + a3[v32];
        v34 = a5[v32++];
        *((_DWORD *)v53 + v33) = v31 + v34;
        v22 = (char *)dest;
      }
      while ( (unsigned int)a4 != v32 );
    }
  }
  else
  {
    if ( 4 * a4 )
    {
      v45 = a6;
      memmove(v22, a3, 4 * a4);
      a6 = v45;
    }
    v39 = 4 * a6;
    if ( v39 )
      memmove(v53, a5, v39);
    v22 = (char *)dest;
  }
  sub_2B38DA0((unsigned int *)a1, (__int64)v22);
  sub_2B3A2E0(a1 + 144, (__int64)v53, (unsigned int)v54, 1u, v35, v36);
  v37 = *(_DWORD *)(a1 + 152);
  if ( v37 )
  {
    v38 = *(_QWORD *)(a1 + 144);
    v47 = *(_DWORD *)(a1 + 152);
    v49[1] = v37;
    v49[0] = v38;
    v48 = (unsigned int **)v49;
    if ( sub_2B397C0(&v48, &v47) )
      *(_DWORD *)(a1 + 152) = 0;
  }
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( dest != v52 )
    _libc_free((unsigned __int64)dest);
}
