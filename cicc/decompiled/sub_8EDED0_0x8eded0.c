// Function: sub_8EDED0
// Address: 0x8eded0
//
__int64 __fastcall sub_8EDED0(unsigned __int8 *a1, char *a2, __int64 a3, _BOOL4 *a4, _DWORD *a5, _QWORD *a6)
{
  unsigned __int8 *v7; // rax
  char v8; // dl
  char *v9; // rcx
  _BYTE *v10; // rbx
  int v11; // eax
  _BOOL4 v12; // edx
  char *v13; // rdi
  char *v14; // rax
  char *v15; // rcx
  char v16; // dl
  char *i; // rax
  __int64 result; // rax
  char *haystack; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23; // [rsp+28h] [rbp-78h]
  __int64 v24; // [rsp+30h] [rbp-70h]
  _BOOL8 v25; // [rsp+38h] [rbp-68h]
  __int64 v26; // [rsp+40h] [rbp-60h]
  __int64 v27; // [rsp+48h] [rbp-58h]
  __int64 v28; // [rsp+50h] [rbp-50h]
  __int64 v29; // [rsp+58h] [rbp-48h]
  __int64 v30; // [rsp+60h] [rbp-40h]
  int v31; // [rsp+68h] [rbp-38h]

  haystack = a2;
  v23 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v24 = a3;
  qword_4F605B8 = 0;
  unk_4F07580 = 1;
  while ( 2 )
  {
    v7 = a1;
    v8 = 95;
    v9 = (char *)&unk_3C1BC41;
    while ( *v7++ == v8 )
    {
      v8 = *v9++;
      if ( !v8 )
      {
        v10 = (_BYTE *)sub_8E9250(a1 + 2, unk_4F605C8, (__int64)&haystack);
        if ( !v25 )
          goto LABEL_10;
        goto LABEL_6;
      }
    }
    v10 = sub_8E9FF0((__int64)a1, 0, 0, 0, 1u, (__int64)&haystack);
    sub_8EB260(a1, 0, 0, (__int64)&haystack);
    if ( !v25 )
      break;
LABEL_6:
    if ( (_DWORD)v29 && !HIDWORD(v29) )
    {
      v23 = 0;
      haystack = a2;
      v24 = a3;
      qword_4F605B8 = 0;
      v25 = 0;
      v26 = 0;
      v27 = 0;
      v28 = 0;
      v29 = 0x100000000LL;
      v30 = 0;
      v31 = 0;
      continue;
    }
    break;
  }
LABEL_10:
  v11 = HIDWORD(v25);
  v12 = 1;
  if ( !HIDWORD(v25) )
  {
    haystack[v23] = 0;
    v13 = haystack;
    while ( 1 )
    {
      v14 = strstr(v13, "::");
      v15 = v14;
      if ( !v14 )
        break;
      while ( 1 )
      {
        v16 = v14[2];
        v13 = v14 + 2;
        for ( i = v14 + 2; v16 == 32; ++i )
          v16 = i[1];
        if ( v16 != 58 || i[1] != 58 )
          break;
        *(_WORD *)v15 = 8224;
        v14 = strstr(v13, "::");
        v15 = v14;
        if ( !v14 )
          goto LABEL_18;
      }
    }
LABEL_18:
    v12 = v25;
    v11 = HIDWORD(v25);
    if ( v10 && !v25 )
      v12 = *v10 != 0;
  }
  *a4 = v12;
  *a5 = v11;
  result = v23 + 1;
  *a6 = v23 + 1;
  return result;
}
