// Function: sub_C823F0
// Address: 0xc823f0
//
__int64 __fastcall sub_C823F0(__int64 a1, char a2)
{
  const char *v3; // rax
  __int64 v4; // rdi
  const char *v5; // r12
  const char *v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned int v10; // r13d
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  struct stat stat_buf; // [rsp+0h] [rbp-150h] BYREF
  _QWORD v16[3]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v17[168]; // [rsp+A8h] [rbp-A8h] BYREF

  v16[0] = v17;
  v16[1] = 0;
  v16[2] = 128;
  v3 = (const char *)sub_CA12A0(a1, v16);
  v4 = 1;
  v5 = v3;
  v6 = v3;
  if ( !__lxstat(1, v3, &stat_buf) )
  {
    v9 = stat_buf.st_mode & 0xF000;
    if ( (_DWORD)v9 != 0x4000 && (stat_buf.st_mode & 0xD000) != 0x8000 )
    {
      v10 = 1;
      sub_2241E50(1, v6, v9, v7, v8);
      goto LABEL_5;
    }
    v4 = (__int64)v5;
    if ( remove(v5) != -1 )
      goto LABEL_12;
  }
  v10 = *__errno_location();
  if ( v10 == 2 && a2 )
  {
LABEL_12:
    v10 = 0;
    sub_2241E40(v4, v6, v12, v13, v14);
  }
  else
  {
    sub_2241E50(v4, v6, v12, v13, v14);
  }
LABEL_5:
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0], v6);
  return v10;
}
