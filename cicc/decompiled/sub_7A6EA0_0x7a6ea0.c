// Function: sub_7A6EA0
// Address: 0x7a6ea0
//
__int64 __fastcall sub_7A6EA0(unsigned __int64 *a1, unsigned __int64 *a2, unsigned int a3)
{
  _BOOL4 v3; // r8d
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 result; // rax

  v3 = 0;
  if ( !*a2 || (result = sub_7A6DD0(a1, a2, dword_4F06BA0 - *a2), v3 = result == 0, (_DWORD)result) )
  {
    v5 = *a1;
    v6 = *a1 % a3;
    if ( v6 )
    {
      result = 0;
      if ( a3 - v6 <= unk_4F06AC0 && v5 <= v6 + unk_4F06AC0 - a3 )
      {
        *a1 = a3 - v6 + v5;
        return 1;
      }
    }
    else
    {
      return !v3;
    }
  }
  return result;
}
