// Function: sub_8257B0
// Address: 0x8257b0
//
char *__fastcall sub_8257B0(char *src)
{
  __int64 v1; // rdx
  __int64 v2; // r8
  __int64 v3; // r9
  char *v4; // r12
  char v5; // dl
  char *i; // rax
  size_t v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+4h] [rbp-3Ch] BYREF
  size_t n[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( sub_5D7700() )
    unk_4F5F774 = 1;
  sub_8EDED0(src, &unk_4F1F760, 0x40000, &v13, &v14, n);
  if ( v13 && (v1 = v14) == 0 )
  {
    v8 = strlen(src);
    v4 = (char *)sub_822B10(v8 + 1, (__int64)&unk_4F1F760, v9, v10, v11, v12);
    strcpy(v4, src);
  }
  else
  {
    v4 = (char *)sub_822B10(n[0], (__int64)&unk_4F1F760, v1, v13, v2, v3);
    if ( v13 )
      sub_8EDED0(src, v4, n[0], &v13, &v14, n);
    else
      memcpy(v4, &unk_4F1F760, n[0]);
  }
  if ( sub_5D7700() )
    unk_4F5F774 = 0;
  v5 = *v4;
  for ( i = v4; v5 == 32; ++i )
    v5 = i[1];
  if ( v5 == 58 && i[1] == 58 )
    return i + 2;
  return v4;
}
