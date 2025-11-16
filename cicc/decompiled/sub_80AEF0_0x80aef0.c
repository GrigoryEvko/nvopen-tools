// Function: sub_80AEF0
// Address: 0x80aef0
//
char *__fastcall sub_80AEF0(__int64 a1)
{
  __int64 v1; // r8
  unsigned __int8 v3; // al
  __int64 v4; // rax
  const char *v5; // rsi
  size_t v6; // rax
  char *v7; // rax
  char *v8; // rax
  __int64 v9; // rcx
  char v10[88]; // [rsp-58h] [rbp-58h] BYREF

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 )
    return *(char **)(a1 + 8);
  v3 = *(_BYTE *)(a1 + 140) - 9;
  if ( *(_QWORD *)a1 )
  {
    if ( v3 > 2u || !*(_BYTE *)(*(_QWORD *)(a1 + 168) + 113LL) )
      return (char *)v1;
    v4 = qword_4F18BC0;
    *(_BYTE *)(a1 + 89) |= 0x48u;
    v5 = "__C%lu";
    qword_4F18BC0 = v4 + 1;
  }
  else
  {
    v9 = qword_4F18BC0;
    *(_BYTE *)(a1 + 89) |= 0x48u;
    v5 = "__C%lu";
    if ( v3 >= 3u )
      v5 = "__E%lu";
    qword_4F18BC0 = v9 + 1;
  }
  sprintf(v10, v5);
  v6 = strlen(v10);
  v7 = (char *)sub_7E1510(v6 + 1);
  v8 = strcpy(v7, v10);
  *(_QWORD *)(a1 + 8) = v8;
  return v8;
}
