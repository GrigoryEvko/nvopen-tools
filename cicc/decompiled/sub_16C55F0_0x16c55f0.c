// Function: sub_16C55F0
// Address: 0x16c55f0
//
__int64 __fastcall sub_16C55F0(__int64 a1, __int64 a2, char a3)
{
  const char *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r8
  unsigned int v8; // ebx
  unsigned __int64 v10[2]; // [rsp+0h] [rbp-140h] BYREF
  _BYTE v11[128]; // [rsp+10h] [rbp-130h] BYREF
  struct stat stat_buf; // [rsp+90h] [rbp-B0h] BYREF

  v10[0] = (unsigned __int64)v11;
  v10[1] = 0x8000000000LL;
  v4 = (const char *)sub_16E32E0(a1, v10);
  if ( a3 )
    v6 = (unsigned int)__xstat(1, v4, &stat_buf);
  else
    v6 = (unsigned int)__lxstat(1, v4, &stat_buf);
  v8 = sub_16C3100(v6, (__int64 *)&stat_buf, a2, v5, v7);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
  return v8;
}
