// Function: sub_7AD240
// Address: 0x7ad240
//
__int64 __fastcall sub_7AD240(int a1, __int16 a2)
{
  int v2; // ebx
  __int16 v3; // r12
  __int64 result; // rax

  v2 = a1;
  if ( dword_4F083D8 )
  {
    v3 = a2;
    if ( a1 )
    {
      do
      {
        result = sub_729660(10);
        --v2;
      }
      while ( v2 );
    }
    if ( a2 )
    {
      do
      {
        result = sub_729660(32);
        --v3;
      }
      while ( v3 );
    }
  }
  else
  {
    result = dword_4D03BA0;
    if ( !dword_4D03BA0 && (a1 || a2) )
      return sub_729660(32);
  }
  return result;
}
