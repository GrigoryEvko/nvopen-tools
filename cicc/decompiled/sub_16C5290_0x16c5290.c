// Function: sub_16C5290
// Address: 0x16c5290
//
__int64 *__fastcall sub_16C5290(__int64 *a1, const char *a2)
{
  const char *v3; // rdi
  ssize_t v4; // rbx
  struct stat *p_stat_buf; // rdx
  size_t v7; // rax
  char *v8; // rdi
  char *v9; // rcx
  char *v10; // rax
  char *v11; // [rsp+0h] [rbp-30E0h]
  char *v12; // [rsp+10h] [rbp-30D0h] BYREF
  __int64 v13; // [rsp+18h] [rbp-30C8h]
  struct stat stat_buf; // [rsp+20h] [rbp-30C0h] BYREF
  char buf[4096]; // [rsp+B0h] [rbp-3030h] BYREF
  struct stat v16; // [rsp+10B0h] [rbp-2030h] BYREF
  char *path[2]; // [rsp+20B0h] [rbp-1030h] BYREF
  _QWORD v18[516]; // [rsp+20C0h] [rbp-1020h] BYREF

  v12 = "/proc/self/exe";
  LOWORD(v18[0]) = 261;
  v13 = 14;
  path[0] = (char *)&v12;
  if ( !(unsigned int)sub_16C51A0((__int64)path, 0) )
  {
    path[0] = (char *)v18;
    if ( v12 )
    {
      sub_16C3260((__int64 *)path, v12, (__int64)&v12[v13]);
      v3 = path[0];
    }
    else
    {
      path[1] = 0;
      v3 = (const char *)v18;
      LOBYTE(v18[0]) = 0;
    }
    v4 = readlink(v3, buf, 0x1000u);
    if ( (_QWORD *)path[0] != v18 )
      j_j___libc_free_0(path[0], v18[0] + 1LL);
    if ( v4 >= 0 )
    {
      *a1 = (__int64)(a1 + 2);
      sub_16C3260(a1, buf, (__int64)&buf[v4]);
      return a1;
    }
    goto LABEL_7;
  }
  if ( *a2 == 47 )
  {
    snprintf((char *)path, 0x1000u, "%s/%s", "/", a2);
    v10 = realpath((const char *)path, buf);
    p_stat_buf = &v16;
    if ( !v10 )
      goto LABEL_7;
    goto LABEL_15;
  }
  if ( !strchr(a2, 47) )
  {
    v8 = getenv("PATH");
    if ( v8 )
    {
      v11 = strdup(v8);
      stat_buf.st_dev = (__dev_t)v11;
      if ( v11 )
      {
        while ( 1 )
        {
          v9 = strsep((char **)&stat_buf, ":");
          if ( !v9 )
            break;
          snprintf((char *)path, 0x1000u, "%s/%s", v9, a2);
          if ( realpath((const char *)path, buf) && !__xstat(1, (const char *)path, &v16) )
          {
            _libc_free((unsigned __int64)v11);
            goto LABEL_16;
          }
        }
        _libc_free((unsigned __int64)v11);
      }
    }
    goto LABEL_7;
  }
  if ( getcwd((char *)&v16, 0x1000u) )
  {
    snprintf((char *)path, 0x1000u, "%s/%s", (const char *)&v16, a2);
    if ( realpath((const char *)path, buf) )
    {
      p_stat_buf = &stat_buf;
LABEL_15:
      if ( !__xstat(1, (const char *)path, p_stat_buf) )
      {
LABEL_16:
        *a1 = (__int64)(a1 + 2);
        v7 = strlen(buf);
        sub_16C3260(a1, buf, (__int64)&buf[v7]);
        return a1;
      }
    }
  }
LABEL_7:
  *a1 = (__int64)(a1 + 2);
  sub_16C3260(a1, byte_3F871B3, (__int64)byte_3F871B3);
  return a1;
}
