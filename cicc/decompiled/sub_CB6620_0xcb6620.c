// Function: sub_CB6620
// Address: 0xcb6620
//
__int64 __fastcall sub_CB6620(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  char *v7; // rdi
  unsigned __int64 v8; // r13
  size_t v9; // r15
  size_t v10; // rax
  bool v11; // cf
  char *i; // rdi
  char *v13; // rax
  char *v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r13d
  int (__fastcall *v17)(__int64, char *, unsigned int); // rax
  int v18; // eax
  size_t v19; // rdx
  char *v20; // rsi
  int (__fastcall *v22)(__int64, char *, unsigned int); // rax
  int v23; // eax
  char *s; // [rsp+10h] [rbp-D0h] BYREF
  size_t v25; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v26; // [rsp+20h] [rbp-C0h]
  _BYTE v27[184]; // [rsp+28h] [rbp-B8h] BYREF

  v6 = a1;
  v7 = *(char **)(a1 + 32);
  v8 = *(_QWORD *)(v6 + 24) - (_QWORD)v7;
  if ( v8 > 3 )
  {
    v22 = *(int (__fastcall **)(__int64, char *, unsigned int))(*(_QWORD *)a2 + 8LL);
    if ( v22 == sub_CB01D0 )
      v23 = snprintf(v7, (unsigned int)v8, *(const char **)(a2 + 8), *(unsigned __int8 *)(a2 + 16));
    else
      v23 = v22(a2, v7, v8);
    v9 = v23 - ((unsigned int)(v23 < (unsigned int)v8) - 1);
    if ( v23 < 0 )
      v9 = (unsigned int)(2 * v8);
    if ( v8 >= v9 )
    {
      *(_QWORD *)(v6 + 32) += v9;
      return v6;
    }
  }
  else
  {
    v9 = 127;
  }
  v25 = 0;
  s = v27;
  v10 = 0;
  v26 = 128;
  v11 = 0;
  if ( !v9 )
    goto LABEL_18;
LABEL_4:
  if ( !v11 )
  {
    if ( v9 > v26 )
    {
      sub_C8D290((__int64)&s, v27, v9, 1u, a5, a6);
      v10 = v25;
    }
    i = s;
    v13 = &s[v10];
    v14 = &s[v9];
    if ( v13 != &s[v9] )
    {
      do
      {
        if ( v13 )
          *v13 = 0;
        ++v13;
      }
      while ( v14 != v13 );
      i = s;
    }
    v15 = *(_QWORD *)a2;
    v25 = v9;
    v16 = v9;
    v17 = *(int (__fastcall **)(__int64, char *, unsigned int))(v15 + 8);
    if ( v17 != sub_CB01D0 )
      goto LABEL_20;
LABEL_13:
    v18 = snprintf(i, v9, *(const char **)(a2 + 8), *(unsigned __int8 *)(a2 + 16));
    goto LABEL_14;
  }
  v25 = v9;
  for ( i = s; ; i = s )
  {
    v16 = v9;
    v17 = *(int (__fastcall **)(__int64, char *, unsigned int))(*(_QWORD *)a2 + 8LL);
    if ( v17 == sub_CB01D0 )
      goto LABEL_13;
LABEL_20:
    v18 = v17(a2, i, v9);
LABEL_14:
    a6 = 2 * v16;
    v19 = v18 - ((unsigned int)(v18 < v16) - 1);
    if ( v18 < 0 )
      v19 = 2 * v16;
    if ( v9 >= v19 )
      break;
    v10 = v25;
    v9 = v19;
    v11 = v19 < v25;
    if ( v19 != v25 )
      goto LABEL_4;
LABEL_18:
    ;
  }
  v20 = s;
  v6 = sub_CB6200(v6, (unsigned __int8 *)s, v19);
  if ( s != v27 )
    _libc_free(s, v20);
  return v6;
}
