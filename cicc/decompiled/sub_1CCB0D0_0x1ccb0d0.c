// Function: sub_1CCB0D0
// Address: 0x1ccb0d0
//
bool __fastcall sub_1CCB0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v7; // cc
  __int64 v8; // rdx
  bool result; // al
  __int64 v10; // r9
  __int64 v11; // r10
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax

  if ( !a3 || !a5 )
    return 0;
  v7 = a3 < a5;
  v8 = a5;
  result = 1;
  if ( !v7 )
    v8 = a3;
  if ( a1 > v8 )
  {
    v10 = a2 % a1 + a1;
    if ( a2 % a1 >= 0 )
      v10 = a2 % a1;
    v11 = a4 % a1 + a1;
    if ( a4 % a1 >= 0 )
      v11 = a4 % a1;
    v12 = (a3 + a2) % a1;
    v13 = a5 + a4;
    v14 = v12 + a1;
    if ( v12 >= 0 )
      v14 = v12;
    v15 = v13 % a1;
    v16 = v13 % a1 + a1;
    if ( v15 < 0 )
      v15 = v16;
    if ( v14 <= v10 || v11 >= v15 )
    {
      result = v14 < v10 && v11 < v15;
      if ( result )
      {
        if ( a1 <= v11 || a1 == v10 || v15 <= v10 )
          return v15 > 0 && v11 < v14 && v14 != 0;
      }
      else
      {
        result = v14 > v10 && v11 > v15;
        if ( result )
        {
          if ( a1 == v11 || a1 <= v10 || v11 >= v14 )
            return v14 > 0 && v15 > v10 && v15 != 0;
        }
        else
        {
          return 1;
        }
      }
    }
    else
    {
      return v11 < v14 && v15 > v10;
    }
  }
  return result;
}
