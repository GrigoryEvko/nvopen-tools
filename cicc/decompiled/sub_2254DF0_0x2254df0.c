// Function: sub_2254DF0
// Address: 0x2254df0
//
__int64 __fastcall sub_2254DF0(__int64 a1, mbstate_t *a2, void *a3, unsigned __int64 a4, size_t a5)
{
  mbstate_t v8; // rax
  void *v9; // r15
  void *v10; // rsp
  bool v11; // cc
  unsigned int v12; // r12d
  size_t v13; // r13
  char *v14; // rax
  char *v15; // r10
  size_t v16; // rax
  char *v17; // rcx
  size_t v19; // rbx
  mbstate_t v20; // rax
  const char *v21; // rsi
  size_t v22; // rax
  __int64 v23; // [rsp+0h] [rbp-70h] BYREF
  const char **p_s; // [rsp+8h] [rbp-68h]
  wchar_t *dst; // [rsp+10h] [rbp-60h]
  char *v26; // [rsp+18h] [rbp-58h]
  unsigned __int64 v27; // [rsp+20h] [rbp-50h]
  void *s; // [rsp+28h] [rbp-48h] BYREF
  mbstate_t p; // [rsp+38h] [rbp-38h] BYREF

  v8 = *a2;
  s = a3;
  v27 = a4;
  p = v8;
  v9 = a3;
  v23 = __uselocale();
  v10 = alloca(4 * a5 + 8);
  v11 = a4 <= (unsigned __int64)s;
  v12 = 0;
  dst = (wchar_t *)&v23;
  if ( !v11 && a5 )
  {
    p_s = (const char **)&s;
    while ( 1 )
    {
      v13 = v27 - (_QWORD)v9;
      v14 = (char *)memchr(v9, 0, v27 - (_QWORD)v9);
      v15 = v14;
      if ( v14 )
        v13 = v14 - (_BYTE *)v9;
      else
        v15 = (char *)v27;
      v26 = v15;
      v16 = mbsnrtowcs(dst, p_s, v13, a5, a2);
      if ( v16 == -1 )
        break;
      v17 = (char *)s;
      if ( s )
      {
        LODWORD(v13) = (_DWORD)s - (_DWORD)v9;
      }
      else
      {
        s = v26;
        v17 = v26;
      }
      v12 += v13;
      if ( v27 > (unsigned __int64)v17 )
      {
        v19 = a5 - v16;
        if ( v19 )
        {
          v20 = *a2;
          v9 = v17 + 1;
          ++v12;
          a5 = v19 - 1;
          s = v17 + 1;
          p = v20;
          if ( (unsigned __int64)(v17 + 1) < v27 )
          {
            if ( a5 )
              continue;
          }
        }
      }
      goto LABEL_10;
    }
    s = v9;
    v21 = (const char *)v9;
    while ( 1 )
    {
      v22 = mbrtowc(0, v21, v27 - (_QWORD)v21, &p);
      if ( v22 > 0xFFFFFFFFFFFFFFFDLL )
        break;
      v21 = (char *)s + v22;
      s = (char *)s + v22;
    }
    v12 = (_DWORD)s + v12 - (_DWORD)v9;
    *a2 = p;
  }
LABEL_10:
  __uselocale();
  return v12;
}
