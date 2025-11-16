// Function: sub_16F6380
// Address: 0x16f6380
//
char *__fastcall sub_16F6380(__int64 a1, char *a2)
{
  char *v3; // rsi
  char v4; // al
  unsigned __int64 v6; // rax
  __int64 v7; // r10

  v3 = *(char **)(a1 + 48);
  if ( v3 == a2 )
    return a2;
  v4 = *a2;
  if ( (unsigned __int8)(*a2 - 32) <= 0x5Eu || v4 == 9 )
    return a2 + 1;
  if ( v4 >= 0 )
    return a2;
  v6 = sub_16F61C0(a2, v3 - a2);
  if ( HIDWORD(v6)
    && (_DWORD)v6 != 65279
    && ((unsigned int)(v6 - 160) <= 0xD75F
     || (_DWORD)v6 == 133
     || (unsigned int)(v6 - 57344) <= 0x1FFD
     || (unsigned int)(v6 - 0x10000) <= 0xFFFFF) )
  {
    return (char *)(v7 + HIDWORD(v6));
  }
  else
  {
    return (char *)v7;
  }
}
