// Function: sub_C84AB0
// Address: 0xc84ab0
//
__int64 __fastcall sub_C84AB0(char **a1, char *a2, size_t a3, const void *a4, size_t a5, unsigned int a6)
{
  unsigned int v8; // r15d
  char *v10; // rbx
  char *v11; // r15
  char v12; // al
  char v13; // dl
  char *s1; // [rsp+8h] [rbp-1A8h]
  size_t v16; // [rsp+10h] [rbp-1A0h]
  char *v17; // [rsp+18h] [rbp-198h]
  bool v19; // [rsp+2Fh] [rbp-181h]
  _QWORD v20[4]; // [rsp+30h] [rbp-180h] BYREF
  __int16 v21; // [rsp+50h] [rbp-160h]
  _QWORD s2[3]; // [rsp+60h] [rbp-150h] BYREF
  _BYTE v23[312]; // [rsp+78h] [rbp-138h] BYREF

  if ( a3 )
  {
    v16 = (size_t)a1[1];
    s1 = *a1;
    if ( a6 > 1 )
    {
      if ( v16 >= a3 )
      {
        v10 = a2;
        v11 = *a1;
        v17 = &s1[a3];
        while ( 1 )
        {
          v19 = sub_C80220(*v11, a6);
          if ( v19 != sub_C80220(*v10, a6) )
            break;
          if ( !v19 )
          {
            v12 = *v11;
            if ( (unsigned __int8)(*v11 - 65) < 0x1Au )
              v12 = *v11 + 32;
            v13 = *v10;
            if ( (unsigned __int8)(*v10 - 65) < 0x1Au )
              v13 = *v10 + 32;
            if ( v13 != v12 )
              break;
          }
          ++v11;
          ++v10;
          if ( v17 == v11 )
            goto LABEL_12;
        }
      }
    }
    else if ( v16 >= a3 && !memcmp(s1, a2, a3) )
    {
      goto LABEL_12;
    }
    return 0;
  }
  v8 = 0;
  if ( !a5 )
    return v8;
  v16 = (size_t)a1[1];
  s1 = *a1;
  if ( a6 <= 1 )
    goto LABEL_4;
LABEL_12:
  if ( a5 != a3 )
  {
LABEL_4:
    s2[1] = 0;
    v20[0] = a4;
    s2[0] = v23;
    v20[1] = a5;
    s2[2] = 256;
    v20[2] = &s1[a3];
    v20[3] = v16 - a3;
    v21 = 1285;
    sub_CA0EC0(v20, s2);
    sub_C844F0(a1, s2);
    if ( (_BYTE *)s2[0] != v23 )
      _libc_free(s2[0], s2);
    return 1;
  }
  v8 = 1;
  if ( a5 )
    memmove(*a1, a4, a5);
  return v8;
}
