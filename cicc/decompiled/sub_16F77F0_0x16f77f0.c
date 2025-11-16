// Function: sub_16F77F0
// Address: 0x16f77f0
//
char *__fastcall sub_16F77F0(__int64 a1)
{
  char *result; // rax
  unsigned __int64 i; // rcx
  __int64 v3; // rax
  char v4; // dl
  char v5; // dl
  char v6; // dl
  _QWORD v7[5]; // [rsp-28h] [rbp-28h] BYREF

  result = *(char **)(a1 + 40);
  for ( i = *(_QWORD *)(a1 + 48); (char *)i != result; *(_QWORD *)(a1 + 40) = result )
  {
    v4 = *result;
    if ( *result == 37 )
    {
      if ( i > (unsigned __int64)(result + 2) )
      {
        v5 = result[1];
        if ( (unsigned __int8)(v5 - 48) <= 9u || (unsigned __int8)((v5 & 0xDF) - 65) <= 0x19u )
        {
          v6 = result[2];
          if ( (unsigned __int8)(v6 - 48) <= 9u || (unsigned __int8)((v6 & 0xDF) - 65) <= 0x19u )
            goto LABEL_6;
        }
      }
    }
    else if ( (unsigned __int8)(v4 - 97) <= 0x19u || v4 == 45 || (unsigned __int8)(v4 - 65) <= 0x19u )
    {
      goto LABEL_6;
    }
    v7[0] = result;
    v7[1] = 1;
    result = (char *)sub_16D23E0(v7, "#;/?:@&=+$,_.!~*'()[]", 21, 0);
    if ( result == (char *)-1LL )
      return result;
    i = *(_QWORD *)(a1 + 48);
LABEL_6:
    v3 = *(_QWORD *)(a1 + 40);
    ++*(_DWORD *)(a1 + 60);
    result = (char *)(v3 + 1);
  }
  return result;
}
