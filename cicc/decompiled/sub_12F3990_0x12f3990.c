// Function: sub_12F3990
// Address: 0x12f3990
//
__int64 __fastcall sub_12F3990(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _BYTE *a7,
        __int64 a8)
{
  _QWORD *v10; // rsi
  _QWORD *v11; // rdi
  int *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  _DWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  int v21; // r13d
  __int64 v22; // rax
  _DWORD *v23; // r8
  __int64 v24; // rax
  _QWORD *v25; // rsi
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  int v31; // edi
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rdx
  _QWORD *v35; // rdi
  size_t v37; // rdx
  __int64 v38; // [rsp+8h] [rbp-68h] BYREF
  void *dest; // [rsp+10h] [rbp-60h]
  size_t v40; // [rsp+18h] [rbp-58h]
  _QWORD v41[2]; // [rsp+20h] [rbp-50h] BYREF
  int *v42; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]
  _QWORD src[6]; // [rsp+40h] [rbp-30h] BYREF

  v10 = a7;
  dest = v41;
  v40 = 0;
  LOBYTE(v41[0]) = 0;
  if ( !a7 )
  {
    LOBYTE(src[0]) = 0;
    v37 = 0;
    v11 = v41;
    v42 = (int *)src;
LABEL_47:
    v40 = v37;
    *((_BYTE *)v11 + v37) = 0;
    v12 = v42;
    goto LABEL_6;
  }
  v42 = (int *)src;
  sub_12EFD20((__int64 *)&v42, a7, (__int64)&a7[a8]);
  v11 = dest;
  v12 = (int *)dest;
  if ( v42 == (int *)src )
  {
    v37 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)dest = src[0];
      }
      else
      {
        v10 = src;
        memcpy(dest, src, n);
      }
      v37 = n;
      v11 = dest;
    }
    goto LABEL_47;
  }
  a4 = src[0];
  v10 = (_QWORD *)n;
  if ( dest == v41 )
  {
    dest = v42;
    v40 = n;
    v41[0] = src[0];
  }
  else
  {
    v13 = v41[0];
    dest = v42;
    v40 = n;
    v41[0] = src[0];
    if ( v12 )
    {
      v42 = v12;
      src[0] = v13;
      goto LABEL_6;
    }
  }
  v42 = (int *)src;
  v12 = (int *)src;
LABEL_6:
  n = 0;
  *(_BYTE *)v12 = 0;
  if ( v42 != (int *)src )
  {
    v10 = (_QWORD *)(src[0] + 1LL);
    j_j___libc_free_0(v42, src[0] + 1LL);
  }
  v14 = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 168) = v14 - 32;
  v15 = *(_DWORD **)(v14 - 32);
  v16 = v14 - 16;
  if ( v15 != (_DWORD *)(v14 - 16) )
  {
    v10 = (_QWORD *)(*(_QWORD *)(v14 - 16) + 1LL);
    j_j___libc_free_0(v15, v10);
  }
  *(_QWORD *)(a1 + 192) -= 4LL;
  *(_DWORD *)(a1 + 16) = a2;
  v19 = sub_16D5D50(v15, v10, v16, a4);
  v20 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v15 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v18 = v20[2];
        v17 = v20[3];
        if ( v19 <= v20[4] )
          break;
        v20 = (_QWORD *)v20[3];
        if ( !v17 )
          goto LABEL_15;
      }
      v15 = v20;
      v20 = (_QWORD *)v20[2];
    }
    while ( v18 );
LABEL_15:
    v21 = -1;
    if ( v15 != dword_4FA0208 && v19 >= *((_QWORD *)v15 + 4) )
    {
      v22 = *((_QWORD *)v15 + 7);
      v23 = v15 + 12;
      if ( v22 )
      {
        v19 = *(unsigned int *)(a1 + 8);
        v15 += 12;
        do
        {
          while ( 1 )
          {
            v18 = *(_QWORD *)(v22 + 16);
            v17 = *(_QWORD *)(v22 + 24);
            if ( *(_DWORD *)(v22 + 32) >= (int)v19 )
              break;
            v22 = *(_QWORD *)(v22 + 24);
            if ( !v17 )
              goto LABEL_22;
          }
          v15 = (_DWORD *)v22;
          v22 = *(_QWORD *)(v22 + 16);
        }
        while ( v18 );
LABEL_22:
        v21 = -1;
        if ( v23 != v15 && (int)v19 >= v15[8] )
          v21 = v15[9] - 1;
      }
    }
  }
  else
  {
    v21 = -1;
  }
  v24 = sub_16D5D50(v15, v19, v17, v18);
  v25 = dword_4FA0208;
  v38 = v24;
  v26 = v24;
  v27 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_32;
  do
  {
    while ( 1 )
    {
      v28 = v27[2];
      v29 = v27[3];
      if ( v26 <= v27[4] )
        break;
      v27 = (_QWORD *)v27[3];
      if ( !v29 )
        goto LABEL_30;
    }
    v25 = v27;
    v27 = (_QWORD *)v27[2];
  }
  while ( v28 );
LABEL_30:
  if ( v25 == (_QWORD *)dword_4FA0208 || v26 < v25[4] )
  {
LABEL_32:
    v42 = (int *)&v38;
    v25 = (_QWORD *)sub_12F3810(&qword_4FA0200, v25, (unsigned __int64 **)&v42);
  }
  v30 = v25[7];
  if ( !v30 )
  {
    v32 = (__int64)(v25 + 6);
LABEL_40:
    v42 = (int *)(a1 + 8);
    v32 = sub_12F38E0(v25 + 5, v32, &v42);
    goto LABEL_41;
  }
  v31 = *(_DWORD *)(a1 + 8);
  v32 = (__int64)(v25 + 6);
  do
  {
    while ( 1 )
    {
      v33 = *(_QWORD *)(v30 + 16);
      v34 = *(_QWORD *)(v30 + 24);
      if ( *(_DWORD *)(v30 + 32) >= v31 )
        break;
      v30 = *(_QWORD *)(v30 + 24);
      if ( !v34 )
        goto LABEL_38;
    }
    v32 = v30;
    v30 = *(_QWORD *)(v30 + 16);
  }
  while ( v33 );
LABEL_38:
  if ( v25 + 6 == (_QWORD *)v32 || v31 < *(_DWORD *)(v32 + 32) )
    goto LABEL_40;
LABEL_41:
  v35 = dest;
  *(_DWORD *)(v32 + 36) = v21;
  if ( v35 != v41 )
    j_j___libc_free_0(v35, v41[0] + 1LL);
  return 0;
}
