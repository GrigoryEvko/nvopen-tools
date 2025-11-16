// Function: sub_C825C0
// Address: 0xc825c0
//
__int64 __fastcall sub_C825C0(__int64 a1, unsigned int a2)
{
  const char *v3; // rax
  const char *v4; // r12
  const char *v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // ebx
  struct stat stat_buf; // [rsp+0h] [rbp-150h] BYREF
  _QWORD v13[3]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v14[168]; // [rsp+A8h] [rbp-A8h] BYREF

  v13[0] = v14;
  v13[1] = 0;
  v13[2] = 128;
  v3 = (const char *)sub_CA12A0(a1, v13);
  if ( a2 > 2 )
    BUG();
  v4 = v3;
  v5 = (const char *)dword_3F66FF0[a2];
  v6 = (__int64)v3;
  if ( access(v3, (int)v5) == -1 )
  {
    sub_2241E50(v6, v5, v7, v8, v9);
    v10 = *__errno_location();
  }
  else if ( a2 == 2 && ((v5 = v4, v6 = 1, __xstat(1, v4, &stat_buf)) || (stat_buf.st_mode & 0xF000) != 0x8000) )
  {
    v10 = 13;
    sub_2241E50(1, v4, v7, v8, v9);
  }
  else
  {
    v10 = 0;
    sub_2241E40(v6, v5, v7, v8, v9);
  }
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0], v5);
  return v10;
}
