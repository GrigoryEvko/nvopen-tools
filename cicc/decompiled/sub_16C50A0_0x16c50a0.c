// Function: sub_16C50A0
// Address: 0x16c50a0
//
__int64 __fastcall sub_16C50A0(__int64 a1, char a2)
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
  unsigned __int64 v15[2]; // [rsp+0h] [rbp-140h] BYREF
  _BYTE v16[128]; // [rsp+10h] [rbp-130h] BYREF
  struct stat stat_buf; // [rsp+90h] [rbp-B0h] BYREF

  v15[0] = (unsigned __int64)v16;
  v15[1] = 0x8000000000LL;
  v3 = (const char *)sub_16E32E0(a1, v15);
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
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0]);
  return v10;
}
