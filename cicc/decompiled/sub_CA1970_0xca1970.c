// Function: sub_CA1970
// Address: 0xca1970
//
bool __fastcall sub_CA1970(unsigned int a1)
{
  bool result; // al
  char *v2; // rsi
  __int64 v3; // rax
  __int64 v4; // rdx
  char *v5; // rcx

  result = 1;
  if ( a1 != 173 )
  {
    v2 = (char *)&unk_3F68A40;
    v3 = 711;
    do
    {
      while ( 1 )
      {
        v4 = v3 >> 1;
        v5 = &v2[8 * (v3 >> 1)];
        if ( a1 <= *((_DWORD *)v5 + 1) )
          break;
        v2 = v5 + 8;
        v3 = v3 - v4 - 1;
        if ( v3 <= 0 )
          goto LABEL_7;
      }
      v3 >>= 1;
    }
    while ( v4 > 0 );
LABEL_7:
    result = 0;
    if ( v2 != "InMemoryFileSystem\n" )
      return a1 >= *(_DWORD *)v2;
  }
  return result;
}
