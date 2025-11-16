// Function: sub_CA7D50
// Address: 0xca7d50
//
char *__fastcall sub_CA7D50(__int64 a1)
{
  char *result; // rax
  unsigned __int64 i; // rcx
  __int64 v3; // rax
  char v4; // dl
  _QWORD v5[5]; // [rsp-28h] [rbp-28h] BYREF

  result = *(char **)(a1 + 40);
  for ( i = *(_QWORD *)(a1 + 48); (char *)i != result; *(_QWORD *)(a1 + 40) = result )
  {
    v4 = *result;
    if ( *result == 37 )
    {
      if ( i > (unsigned __int64)(result + 2)
        && ((unsigned __int8)((result[1] & 0xDF) - 65) <= 0x19u || (unsigned __int8)(result[1] - 48) <= 9u)
        && ((unsigned __int8)((result[2] & 0xDF) - 65) <= 0x19u || (unsigned __int8)(result[2] - 48) <= 9u) )
      {
        goto LABEL_5;
      }
    }
    else if ( v4 == 45 || (unsigned __int8)(v4 - 97) <= 0x19u || (unsigned __int8)(v4 - 65) <= 0x19u )
    {
      goto LABEL_5;
    }
    v5[0] = result;
    v5[1] = 1;
    result = (char *)sub_C934D0(v5, "#;/?:@&=+$,_.!~*'()[]", 21, 0);
    if ( result == (char *)-1LL )
      return result;
    i = *(_QWORD *)(a1 + 48);
LABEL_5:
    v3 = *(_QWORD *)(a1 + 40);
    ++*(_DWORD *)(a1 + 60);
    result = (char *)(v3 + 1);
  }
  return result;
}
