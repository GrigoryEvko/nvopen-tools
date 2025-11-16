// Function: sub_B4EEA0
// Address: 0xb4eea0
//
__int64 __fastcall sub_B4EEA0(int *a1, __int64 a2, int a3)
{
  __int64 i; // rax
  int v8; // edx

  if ( a3 != a2 )
    return 0;
  if ( !(unsigned __int8)sub_B4ED30(a1, (unsigned int)a3, a3) )
  {
    if ( a3 <= 0 )
      return 1;
    for ( i = 0; ; ++i )
    {
      v8 = a1[i];
      if ( v8 != -1 && v8 != (_DWORD)i && v8 != a3 + (_DWORD)i )
        break;
      if ( a3 - 1 == i )
        return 1;
    }
  }
  return 0;
}
