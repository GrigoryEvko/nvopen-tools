// Function: sub_C9C600
// Address: 0xc9c600
//
__int64 *__fastcall sub_C9C600(__int64 *a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *v5; // r15
  __int64 v8; // r8
  size_t v9; // rdx
  unsigned int v10; // r15d
  _BYTE *v11; // rdi
  size_t v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // rdi
  size_t v16; // [rsp+8h] [rbp-138h]
  __int64 v17; // [rsp+8h] [rbp-138h]
  __int64 v18; // [rsp+10h] [rbp-130h] BYREF
  __int64 v19; // [rsp+18h] [rbp-128h]
  void *dest; // [rsp+20h] [rbp-120h] BYREF
  size_t v21; // [rsp+28h] [rbp-118h]
  _QWORD v22[2]; // [rsp+30h] [rbp-110h] BYREF
  _QWORD v23[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v24; // [rsp+50h] [rbp-F0h] BYREF
  _QWORD v25[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v26; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v27; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v28; // [rsp+A0h] [rbp-A0h]
  _QWORD *v29; // [rsp+B0h] [rbp-90h] BYREF
  size_t n; // [rsp+B8h] [rbp-88h]
  _QWORD src[16]; // [rsp+C0h] [rbp-80h] BYREF

  v5 = (_BYTE *)a4;
  if ( a2 )
  {
    dest = v22;
    sub_C95DE0((__int64 *)&dest, a2, (__int64)&a2[a3]);
    v9 = v21;
    if ( v21 )
      goto LABEL_3;
  }
  else
  {
    LOBYTE(v22[0]) = 0;
    dest = v22;
    v21 = 0;
  }
  if ( a5 == 1 )
  {
    if ( *v5 == 45 )
    {
      v29 = src;
      sub_C95DE0((__int64 *)&v29, "out", (__int64)"");
      goto LABEL_12;
    }
  }
  else if ( !v5 )
  {
    LOBYTE(src[0]) = 0;
    v11 = dest;
    v12 = 0;
    v29 = src;
LABEL_26:
    v21 = v12;
    v11[v12] = 0;
    v13 = v29;
    goto LABEL_16;
  }
  v29 = src;
  sub_C95DE0((__int64 *)&v29, v5, (__int64)&v5[a5]);
LABEL_12:
  v11 = dest;
  a4 = (__int64)v29;
  v12 = n;
  v13 = dest;
  if ( v29 == src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v12 = n;
      v11 = dest;
    }
    goto LABEL_26;
  }
  if ( dest == v22 )
  {
    dest = v29;
    v21 = n;
    v22[0] = src[0];
    goto LABEL_29;
  }
  v14 = v22[0];
  dest = v29;
  v21 = n;
  v22[0] = src[0];
  if ( !v13 )
  {
LABEL_29:
    v29 = src;
    v13 = src;
    goto LABEL_16;
  }
  v29 = v13;
  src[0] = v14;
LABEL_16:
  n = 0;
  *(_BYTE *)v13 = 0;
  if ( v29 != src )
    j_j___libc_free_0(v29, src[0] + 1LL);
  if ( 0x3FFFFFFFFFFFFFFFLL - v21 <= 0xA )
    sub_4262D8((__int64)"basic_string::append");
  a2 = ".time-trace";
  sub_2241490(&dest, ".time-trace", 11, a4);
  v9 = v21;
LABEL_3:
  v16 = v9;
  LODWORD(v18) = 0;
  v19 = sub_2241E40(&dest, a2, v9, a4, v8);
  sub_CB7060(&v29, dest, v16, &v18, 7);
  if ( (_DWORD)v18 )
  {
    sub_8FD6D0((__int64)v23, "Could not open ", &dest);
    v27 = v23;
    v28 = 260;
    v10 = v18;
    v17 = v19;
    sub_CA0F50(v25, &v27);
    sub_C63F00(a1, (__int64)v25, v10, v17);
    if ( (__int64 *)v25[0] != &v26 )
      j_j___libc_free_0(v25[0], v26 + 1);
    if ( (__int64 *)v23[0] != &v24 )
      j_j___libc_free_0(v23[0], v24 + 1);
  }
  else
  {
    sub_C9C5C0((__int64)&v29);
    *a1 = 1;
  }
  sub_CB5B00(&v29);
  if ( dest != v22 )
    j_j___libc_free_0(dest, v22[0] + 1LL);
  return a1;
}
