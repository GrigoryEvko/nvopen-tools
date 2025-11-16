// Function: sub_C824F0
// Address: 0xc824f0
//
__int64 __fastcall sub_C824F0(__int64 a1, bool *a2)
{
  struct statfs *p_buf; // rsi
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  char *v8; // rdi
  int v9; // ebx
  bool v11; // al
  char *file; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+10h] [rbp-A0h] BYREF
  struct statfs buf; // [rsp+20h] [rbp-90h] BYREF

  sub_CA0F50(&file, a1);
  p_buf = &buf;
  v4 = statfs(file, &buf);
  v8 = file;
  v9 = v4;
  if ( file != (char *)&v13 )
  {
    p_buf = (struct statfs *)(v13 + 1);
    j_j___libc_free_0(file, v13 + 1);
  }
  if ( v9 )
  {
    sub_2241E50(v8, p_buf, v5, v6, v7);
    return (unsigned int)*__errno_location();
  }
  else
  {
    v11 = LODWORD(buf.f_type) != 26985 && LODWORD(buf.f_type) != -11317950 && LODWORD(buf.f_type) != 20859;
    *a2 = v11;
    sub_2241E40(v8, p_buf, v5, v6, v7);
    return 0;
  }
}
