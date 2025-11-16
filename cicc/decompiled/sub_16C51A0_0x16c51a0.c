// Function: sub_16C51A0
// Address: 0x16c51a0
//
__int64 __fastcall sub_16C51A0(__int64 a1, int a2)
{
  const char *v3; // r13
  const char *v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // r13d
  unsigned __int64 v11[2]; // [rsp+0h] [rbp-140h] BYREF
  _BYTE v12[128]; // [rsp+10h] [rbp-130h] BYREF
  struct stat stat_buf; // [rsp+90h] [rbp-B0h] BYREF

  v11[0] = (unsigned __int64)v12;
  v11[1] = 0x8000000000LL;
  v3 = (const char *)sub_16E32E0(a1, v11);
  v4 = (const char *)dword_42AEF30[a2];
  v5 = (__int64)v3;
  if ( access(v3, (int)v4) == -1 )
  {
    sub_2241E50(v3, v4, v6, v7, v8);
    v9 = *__errno_location();
  }
  else if ( a2 == 2 && ((v4 = v3, v5 = 1, __xstat(1, v3, &stat_buf)) || (stat_buf.st_mode & 0xF000) != 0x8000) )
  {
    v9 = 13;
    sub_2241E50(1, v4, v6, v7, v8);
  }
  else
  {
    v9 = 0;
    sub_2241E40(v5, v4, v6, v7, v8);
  }
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0]);
  return v9;
}
