// Function: sub_2207AC0
// Address: 0x2207ac0
//
void __noreturn sub_2207AC0()
{
  __int64 v0; // rax
  _BYTE *v1; // rbp
  bool v2; // zf
  unsigned __int8 *v3; // rbp
  char *v4; // r12
  int v5[7]; // [rsp+Ch] [rbp-1Ch] BYREF

  if ( !byte_4FD43F8 )
  {
    byte_4FD43F8 = 1;
    v0 = sub_2253520();
    if ( v0 )
    {
      v1 = *(_BYTE **)(v0 + 8);
      v2 = *v1 == 42;
      v5[0] = -1;
      v3 = &v1[v2];
      v4 = sub_8EE130(v3, 0, 0, v5);
      fwrite("terminate called after throwing an instance of '", 1u, 0x30u, stderr);
      if ( v5[0] )
        fputs((const char *)v3, stderr);
      else
        fputs(v4, stderr);
      fwrite("'\n", 1u, 2u, stderr);
      if ( !v5[0] )
        _libc_free((unsigned __int64)v4);
      sub_22534D0();
    }
    fwrite("terminate called without an active exception\n", 1u, 0x2Du, stderr);
    abort();
  }
  fwrite("terminate called recursively\n", 1u, 0x1Du, stderr);
  abort();
}
