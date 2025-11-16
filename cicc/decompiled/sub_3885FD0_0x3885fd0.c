// Function: sub_3885FD0
// Address: 0x3885fd0
//
__int64 __fastcall sub_3885FD0(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 *a4)
{
  __int64 result; // rax
  _BYTE *v6; // r10
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r10d
  __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // r8d
  unsigned __int64 v14; // rsi
  const char *v15; // [rsp+0h] [rbp-30h] BYREF
  char v16; // [rsp+10h] [rbp-20h]
  char v17; // [rsp+11h] [rbp-1Fh]

  result = a3 - a2;
  *a4 = 0;
  if ( a3 - a2 > 15 )
  {
    v6 = a2 + 16;
    result = 0;
    do
    {
      while ( 1 )
      {
        v8 = 16 * result;
        *a4 = v8;
        v9 = (char)*a2;
        if ( (unsigned __int8)(*a2 - 48) > 9u )
          break;
        ++a2;
        result = (unsigned int)(char)(v9 - 48) + v8;
        *a4 = result;
        if ( a2 == v6 )
          goto LABEL_9;
      }
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
      result = v7 + v8;
      ++a2;
      *a4 = result;
    }
    while ( a2 != v6 );
  }
LABEL_9:
  a4[1] = 0;
  if ( a2 != a3 )
  {
    result = 0;
    v10 = 0;
    while ( 1 )
    {
      v12 = 16 * result;
      a4[1] = v12;
      v13 = (char)*a2;
      if ( (unsigned __int8)(*a2 - 48) > 9u )
      {
        if ( (unsigned __int8)(v13 - 97) <= 5u )
        {
          v11 = (unsigned int)(v13 - 87);
        }
        else
        {
          v11 = 0xFFFFFFFFLL;
          if ( (unsigned __int8)(v13 - 65) <= 5u )
            v11 = (unsigned int)(v13 - 55);
        }
      }
      else
      {
        v11 = (unsigned int)(char)(v13 - 48);
      }
      result = v11 + v12;
      ++v10;
      ++a2;
      a4[1] = result;
      if ( v10 > 15 )
        break;
      if ( a3 == a2 )
        return result;
    }
    if ( a3 != a2 )
    {
      v14 = *(_QWORD *)(a1 + 48);
      v17 = 1;
      v15 = "constant bigger than 128 bits detected!";
      v16 = 3;
      return sub_38814C0(a1, v14, (__int64)&v15);
    }
  }
  return result;
}
