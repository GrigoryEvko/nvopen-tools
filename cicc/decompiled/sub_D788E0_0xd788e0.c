// Function: sub_D788E0
// Address: 0xd788e0
//
__int64 __fastcall sub_D788E0(unsigned __int64 a1, __int16 a2, unsigned __int64 a3, __int16 a4)
{
  __int64 result; // rax
  int v7; // r15d
  int v8; // eax
  int v9; // [rsp+Ch] [rbp-34h]

  if ( !a1 )
    return (unsigned int)-(a3 != 0);
  result = 1;
  if ( a3 )
  {
    v7 = a4;
    v9 = sub_D788C0(a1, a2);
    v8 = sub_D788C0(a3, v7);
    if ( v9 == v8 )
    {
      if ( a2 >= a4 )
        return (unsigned int)-sub_F042F0(a3, a1, (unsigned int)(a2 - v7));
      else
        return sub_F042F0(a1, a3, (unsigned int)(v7 - a2));
    }
    else
    {
      return 2 * (unsigned int)(v9 >= v8) - 1;
    }
  }
  return result;
}
