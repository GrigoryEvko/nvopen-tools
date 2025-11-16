// Function: sub_16E8450
// Address: 0x16e8450
//
__int64 __fastcall sub_16E8450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  char *v7; // rdi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rax
  char *v11; // rdi
  int (__fastcall *v12)(__int64, char *, unsigned int); // rax
  int v13; // eax
  size_t v14; // rdx
  char *v15; // rax
  char *v16; // rdx
  int (__fastcall *v18)(__int64, char *, unsigned int); // rax
  int v19; // eax
  char *s; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-B8h]
  _BYTE v22[176]; // [rsp+20h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = *(char **)(a1 + 24);
  v8 = *(_QWORD *)(v6 + 16) - (_QWORD)v7;
  if ( v8 > 3 )
  {
    v18 = *(int (__fastcall **)(__int64, char *, unsigned int))(*(_QWORD *)a2 + 8LL);
    if ( v18 == sub_16C1080 )
      v19 = snprintf(v7, (unsigned int)v8, *(const char **)(a2 + 8), *(unsigned __int8 *)(a2 + 16));
    else
      v19 = v18(a2, v7, v8);
    v9 = v19 - ((unsigned int)(v19 < (unsigned int)v8) - 1);
    if ( v19 < 0 )
      v9 = (unsigned int)(2 * v8);
    if ( v8 >= v9 )
    {
      *(_QWORD *)(v6 + 24) += v9;
      return v6;
    }
  }
  else
  {
    v9 = 127;
  }
  s = v22;
  v21 = 0x8000000000LL;
  v10 = 0;
  while ( 1 )
  {
    if ( v9 < v10 )
    {
      LODWORD(v21) = v9;
      v11 = s;
      goto LABEL_5;
    }
    if ( v9 > v10 )
    {
      if ( v9 > HIDWORD(v21) )
      {
        sub_16CD150((__int64)&s, v22, v9, 1, a5, a6);
        v10 = (unsigned int)v21;
      }
      v11 = s;
      v15 = &s[v10];
      v16 = &s[v9];
      if ( v15 != &s[v9] )
      {
        do
        {
          if ( v15 )
            *v15 = 0;
          ++v15;
        }
        while ( v16 != v15 );
        v11 = s;
      }
      LODWORD(v21) = v9;
LABEL_5:
      v12 = *(int (__fastcall **)(__int64, char *, unsigned int))(*(_QWORD *)a2 + 8LL);
      if ( v12 == sub_16C1080 )
        goto LABEL_6;
      goto LABEL_14;
    }
    v11 = s;
    v12 = *(int (__fastcall **)(__int64, char *, unsigned int))(*(_QWORD *)a2 + 8LL);
    if ( v12 == sub_16C1080 )
    {
LABEL_6:
      v13 = snprintf(v11, v9, *(const char **)(a2 + 8), *(unsigned __int8 *)(a2 + 16));
      goto LABEL_7;
    }
LABEL_14:
    v13 = v12(a2, v11, v9);
LABEL_7:
    a6 = 2 * v9;
    v14 = v13 - ((unsigned int)(v13 < (unsigned int)v9) - 1);
    if ( v13 < 0 )
      v14 = (unsigned int)(2 * v9);
    if ( v9 >= v14 )
      break;
    v10 = (unsigned int)v21;
    v9 = v14;
  }
  v6 = sub_16E7EE0(v6, s, v14);
  if ( s != v22 )
    _libc_free((unsigned __int64)s);
  return v6;
}
