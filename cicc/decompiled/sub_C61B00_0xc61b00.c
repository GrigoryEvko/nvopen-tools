// Function: sub_C61B00
// Address: 0xc61b00
//
__int64 __fastcall sub_C61B00(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  size_t v10; // rsi
  _QWORD *v11; // rdi
  int *v12; // rax
  __int64 v13; // rdi
  int *v14; // rdi
  _QWORD *v15; // r13
  int *v16; // r14
  unsigned __int64 v17; // rdx
  int *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  int v23; // r13d
  _QWORD *v24; // r14
  _QWORD *v25; // rax
  _QWORD *v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rax
  int v30; // edi
  __int64 v31; // r8
  __int64 v32; // rcx
  __int64 v33; // rdx
  _QWORD *v34; // rdi
  size_t v36; // rdx
  unsigned __int64 v37; // [rsp+8h] [rbp-68h] BYREF
  void *dest; // [rsp+10h] [rbp-60h]
  size_t v39; // [rsp+18h] [rbp-58h]
  _QWORD v40[2]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 *v41; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]
  _QWORD src[6]; // [rsp+40h] [rbp-30h] BYREF

  v10 = (size_t)a7;
  dest = v40;
  v39 = 0;
  LOBYTE(v40[0]) = 0;
  if ( !a7 )
  {
    LOBYTE(src[0]) = 0;
    v36 = 0;
    v11 = v40;
    v41 = src;
LABEL_47:
    v39 = v36;
    *((_BYTE *)v11 + v36) = 0;
    v12 = (int *)v41;
    goto LABEL_6;
  }
  v41 = src;
  sub_C5F830((__int64 *)&v41, a7, (__int64)&a7[a8]);
  v11 = dest;
  v12 = (int *)dest;
  if ( v41 == src )
  {
    v36 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)dest = src[0];
      }
      else
      {
        v10 = (size_t)src;
        memcpy(dest, src, n);
      }
      v36 = n;
      v11 = dest;
    }
    goto LABEL_47;
  }
  v10 = n;
  if ( dest == v40 )
  {
    dest = v41;
    v39 = n;
    v40[0] = src[0];
  }
  else
  {
    v13 = v40[0];
    dest = v41;
    v39 = n;
    v40[0] = src[0];
    if ( v12 )
    {
      v41 = (unsigned __int64 *)v12;
      src[0] = v13;
      goto LABEL_6;
    }
  }
  v41 = src;
  v12 = (int *)src;
LABEL_6:
  n = 0;
  *(_BYTE *)v12 = 0;
  v14 = (int *)v41;
  if ( v41 != src )
  {
    v10 = src[0] + 1LL;
    j_j___libc_free_0(v41, src[0] + 1LL);
  }
  *(_QWORD *)(a1 + 184) -= 4LL;
  *(_WORD *)(a1 + 14) = a2;
  v15 = sub_C52410();
  v16 = (int *)(v15 + 1);
  v17 = sub_C959E0(v14, v10);
  v18 = (int *)v15[2];
  if ( v18 )
  {
    v14 = (int *)(v15 + 1);
    do
    {
      while ( 1 )
      {
        v10 = *((_QWORD *)v18 + 2);
        v19 = *((_QWORD *)v18 + 3);
        if ( v17 <= *((_QWORD *)v18 + 4) )
          break;
        v18 = (int *)*((_QWORD *)v18 + 3);
        if ( !v19 )
          goto LABEL_13;
      }
      v14 = v18;
      v18 = (int *)*((_QWORD *)v18 + 2);
    }
    while ( v10 );
LABEL_13:
    if ( v16 != v14 && v17 >= *((_QWORD *)v14 + 4) )
      v16 = v14;
  }
  if ( v16 == (int *)((char *)sub_C52410() + 8) || (v20 = *((_QWORD *)v16 + 7)) == 0 )
  {
    v23 = -1;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 8);
    v14 = v16 + 12;
    do
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v20 + 16);
        v22 = *(_QWORD *)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) >= (int)v10 )
          break;
        v20 = *(_QWORD *)(v20 + 24);
        if ( !v22 )
          goto LABEL_22;
      }
      v14 = (int *)v20;
      v20 = *(_QWORD *)(v20 + 16);
    }
    while ( v21 );
LABEL_22:
    v23 = -1;
    if ( v16 + 12 != v14 && (int)v10 >= v14[8] )
      v23 = v14[9] - 1;
  }
  v24 = sub_C52410();
  v37 = sub_C959E0(v14, v10);
  v25 = (_QWORD *)v24[2];
  v26 = v24 + 1;
  if ( !v25 )
    goto LABEL_32;
  do
  {
    while ( 1 )
    {
      v27 = v25[2];
      v28 = v25[3];
      if ( v37 <= v25[4] )
        break;
      v25 = (_QWORD *)v25[3];
      if ( !v28 )
        goto LABEL_30;
    }
    v26 = v25;
    v25 = (_QWORD *)v25[2];
  }
  while ( v27 );
LABEL_30:
  if ( v24 + 1 == v26 || v37 < v26[4] )
  {
LABEL_32:
    v41 = &v37;
    v26 = (_QWORD *)sub_C61980(v24, v26, &v41);
  }
  v29 = v26[7];
  if ( !v29 )
  {
    v31 = (__int64)(v26 + 6);
LABEL_40:
    v41 = (unsigned __int64 *)(a1 + 8);
    v31 = sub_C61A50(v26 + 5, v31, (int **)&v41);
    goto LABEL_41;
  }
  v30 = *(_DWORD *)(a1 + 8);
  v31 = (__int64)(v26 + 6);
  do
  {
    while ( 1 )
    {
      v32 = *(_QWORD *)(v29 + 16);
      v33 = *(_QWORD *)(v29 + 24);
      if ( *(_DWORD *)(v29 + 32) >= v30 )
        break;
      v29 = *(_QWORD *)(v29 + 24);
      if ( !v33 )
        goto LABEL_38;
    }
    v31 = v29;
    v29 = *(_QWORD *)(v29 + 16);
  }
  while ( v32 );
LABEL_38:
  if ( v26 + 6 == (_QWORD *)v31 || v30 < *(_DWORD *)(v31 + 32) )
    goto LABEL_40;
LABEL_41:
  v34 = dest;
  *(_DWORD *)(v31 + 36) = v23;
  if ( v34 != v40 )
    j_j___libc_free_0(v34, v40[0] + 1LL);
  return 0;
}
