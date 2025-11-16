// Function: sub_C822A0
// Address: 0xc822a0
//
__int64 __fastcall sub_C822A0(__int64 a1)
{
  const char *v1; // rdi
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned int v5; // r13d
  _QWORD v7[3]; // [rsp+0h] [rbp-C0h] BYREF
  _BYTE v8[168]; // [rsp+18h] [rbp-A8h] BYREF

  v7[0] = v8;
  v7[1] = 0;
  v7[2] = 128;
  v1 = (const char *)sub_CA12A0(a1, v7);
  if ( chdir(v1) == -1 )
  {
    sub_2241E50(v1, v7, v2, v3, v4);
    v5 = *__errno_location();
  }
  else
  {
    v5 = 0;
    sub_2241E40(v1, v7, v2, v3, v4);
  }
  if ( (_BYTE *)v7[0] != v8 )
    _libc_free(v7[0], v7);
  return v5;
}
