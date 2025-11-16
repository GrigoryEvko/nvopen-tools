// Function: sub_16DD960
// Address: 0x16dd960
//
__int64 *__fastcall sub_16DD960(__int64 *a1, char *a2, __int64 a3, _BYTE *a4, _BYTE *a5)
{
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r8
  size_t v12; // rbx
  _BYTE *v14; // rax
  _BYTE *v15; // rdi
  size_t v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // rdi
  _QWORD *v21; // rdi
  _QWORD *v22; // rdx
  unsigned int v23; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+18h] [rbp-C8h]
  void *dest; // [rsp+20h] [rbp-C0h] BYREF
  size_t v26; // [rsp+28h] [rbp-B8h]
  _QWORD v27[2]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v28; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v29; // [rsp+48h] [rbp-98h]
  _QWORD v30[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v31; // [rsp+60h] [rbp-80h] BYREF
  size_t n; // [rsp+68h] [rbp-78h]
  _QWORD src[14]; // [rsp+70h] [rbp-70h] BYREF

  sub_16D40F0((__int64)&qword_4FA1650);
  if ( a2 )
  {
    dest = v27;
    sub_16D9940((__int64 *)&dest, a2, (__int64)&a2[a3]);
    v12 = v26;
    if ( v26 )
      goto LABEL_3;
  }
  else
  {
    LOBYTE(v27[0]) = 0;
    dest = v27;
    v26 = 0;
  }
  v14 = a5;
  if ( a5 != (_BYTE *)1 )
  {
    if ( !a4 )
    {
      LOBYTE(src[0]) = 0;
      v15 = dest;
      v16 = 0;
      v31 = src;
      goto LABEL_12;
    }
    v28 = a5;
    v31 = src;
    if ( (unsigned __int64)a5 <= 0xF )
    {
      if ( !a5 )
      {
        v22 = src;
        goto LABEL_29;
      }
      v21 = src;
    }
    else
    {
      v31 = (_QWORD *)sub_22409D0(&v31, &v28, 0);
      v21 = v31;
      src[0] = v28;
    }
    memcpy(v21, a4, (size_t)a5);
    v14 = v28;
    v22 = v31;
LABEL_29:
    n = (size_t)v14;
    v14[(_QWORD)v22] = 0;
    goto LABEL_22;
  }
  if ( *a4 != 45 )
  {
    LOBYTE(src[0]) = *a4;
    v31 = src;
    v22 = src;
    goto LABEL_29;
  }
  v31 = src;
  sub_16D9940((__int64 *)&v31, "out", (__int64)"");
LABEL_22:
  v15 = dest;
  v9 = (__int64)v31;
  v16 = n;
  v17 = dest;
  if ( v31 != src )
  {
    if ( dest == v27 )
    {
      dest = v31;
      v26 = n;
      v27[0] = src[0];
    }
    else
    {
      v20 = v27[0];
      dest = v31;
      v26 = n;
      v27[0] = src[0];
      if ( v17 )
      {
        v31 = v17;
        src[0] = v20;
        goto LABEL_13;
      }
    }
    v31 = src;
    v17 = src;
    goto LABEL_13;
  }
  if ( n )
  {
    if ( n == 1 )
      *(_BYTE *)dest = src[0];
    else
      memcpy(dest, src, n);
    v16 = n;
    v15 = dest;
  }
LABEL_12:
  v26 = v16;
  v15[v16] = 0;
  v17 = v31;
LABEL_13:
  n = 0;
  *(_BYTE *)v17 = 0;
  if ( v31 != src )
    j_j___libc_free_0(v31, src[0] + 1LL);
  if ( 0x3FFFFFFFFFFFFFFFLL - v26 <= 0xA )
    goto LABEL_35;
  a2 = ".time-trace";
  sub_2241490(&dest, ".time-trace", 11, v9);
  v12 = v26;
LABEL_3:
  v23 = 0;
  v24 = sub_2241E40(&dest, a2, v10, v9, v11);
  sub_16E8AF0(&v31, dest, v12, &v23, 3);
  if ( !v23 )
  {
    sub_16DD920((__int64)&v31);
    *a1 = 1;
    goto LABEL_5;
  }
  v29 = 0;
  v28 = v30;
  LOBYTE(v30[0]) = 0;
  sub_2240E30(&v28, v26 + 15);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v29) <= 0xE )
LABEL_35:
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v28, "Could not open ", 15, v18);
  sub_2241490(&v28, dest, v26, v19);
  sub_16BCCC0(a1, v23, v24, v28);
  if ( v28 != (_BYTE *)v30 )
    j_j___libc_free_0(v28, v30[0] + 1LL);
LABEL_5:
  sub_16E7C30(&v31);
  if ( dest != v27 )
    j_j___libc_free_0(dest, v27[0] + 1LL);
  return a1;
}
