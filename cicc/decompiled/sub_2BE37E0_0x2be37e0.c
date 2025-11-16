// Function: sub_2BE37E0
// Address: 0x2be37e0
//
bool __fastcall sub_2BE37E0(char *a1, _BYTE *a2, char *a3)
{
  __int64 v5; // rax
  char v6; // si
  __int64 v7; // rdx
  char *v8; // rcx
  bool result; // al

  v5 = a2 - a1;
  if ( a2 - a1 > 0 )
  {
    v6 = *a3;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &a1[v5 >> 1];
        if ( *v8 >= v6 )
          break;
        a1 = v8 + 1;
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          goto LABEL_6;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
  }
LABEL_6:
  result = 0;
  if ( a2 != a1 )
    return *a3 >= *a1;
  return result;
}
