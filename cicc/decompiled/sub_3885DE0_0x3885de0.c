// Function: sub_3885DE0
// Address: 0x3885de0
//
unsigned __int64 __fastcall sub_3885DE0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v3; // rbp
  unsigned __int64 i; // rdi
  __int64 v7; // rdx
  unsigned __int64 result; // rax
  int v9; // ecx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  const char *v12; // [rsp-28h] [rbp-28h] BYREF
  char v13; // [rsp-18h] [rbp-18h]
  char v14; // [rsp-17h] [rbp-17h]
  __int64 v15; // [rsp-8h] [rbp-8h]

  if ( a2 != a3 )
  {
    for ( i = 0; ; i = result )
    {
      v9 = (char)*a2;
      v10 = 16 * i;
      if ( (unsigned __int8)(*a2 - 48) > 9u )
      {
        if ( (unsigned __int8)(v9 - 97) <= 5u )
        {
          v7 = (unsigned int)(v9 - 87);
        }
        else
        {
          v7 = 0xFFFFFFFFLL;
          if ( (unsigned __int8)(v9 - 65) <= 5u )
            v7 = (unsigned int)(v9 - 55);
        }
        result = v7 + v10;
        if ( i > result )
        {
LABEL_11:
          v15 = v3;
          v11 = *(_QWORD *)(a1 + 48);
          v14 = 1;
          v12 = "constant bigger than 64 bits detected!";
          v13 = 3;
          sub_38814C0(a1, v11, (__int64)&v12);
          return 0;
        }
      }
      else
      {
        result = (unsigned int)(char)(v9 - 48) + v10;
        if ( i > result )
          goto LABEL_11;
      }
      if ( a3 == ++a2 )
        return result;
    }
  }
  return 0;
}
