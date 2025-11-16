// Function: sub_302DCD0
// Address: 0x302dcd0
//
void __fastcall sub_302DCD0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 v5; // r8
  size_t v6; // rdx
  __int64 v7; // [rsp+8h] [rbp-98h] BYREF
  void *dest; // [rsp+10h] [rbp-90h]
  size_t v9; // [rsp+18h] [rbp-88h]
  _QWORD v10[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v11; // [rsp+30h] [rbp-70h] BYREF
  size_t n; // [rsp+38h] [rbp-68h]
  _QWORD src[2]; // [rsp+40h] [rbp-60h] BYREF
  void *v14[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v15; // [rsp+70h] [rbp-30h]

  v7 = a2;
  dest = v10;
  v9 = 0;
  LOBYTE(v10[0]) = 0;
  if ( a3 )
    goto LABEL_2;
  v14[0] = &v7;
  v14[2] = " [unsigned LEB]";
  v15 = 779;
  sub_CA0F50((__int64 *)&v11, v14);
  v4 = dest;
  if ( v11 == src )
  {
    v6 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v6 = n;
      v4 = dest;
    }
    v9 = v6;
    *((_BYTE *)v4 + v6) = 0;
    v4 = v11;
    goto LABEL_9;
  }
  if ( dest == v10 )
  {
    dest = v11;
    v9 = n;
    v10[0] = src[0];
  }
  else
  {
    v5 = v10[0];
    dest = v11;
    v9 = n;
    v10[0] = src[0];
    if ( v4 )
    {
      v11 = v4;
      src[0] = v5;
      goto LABEL_9;
    }
  }
  v11 = src;
  v4 = src;
LABEL_9:
  n = 0;
  *(_BYTE *)v4 = 0;
  if ( v11 != src )
    j_j___libc_free_0((unsigned __int64)v11);
LABEL_2:
  sub_31D54C0(a1, v7);
  if ( dest != v10 )
    j_j___libc_free_0((unsigned __int64)dest);
}
