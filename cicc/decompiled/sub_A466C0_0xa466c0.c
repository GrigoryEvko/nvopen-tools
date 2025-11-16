// Function: sub_A466C0
// Address: 0xa466c0
//
__int64 __fastcall sub_A466C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // r12d
  unsigned int i; // ebx
  __int64 v6; // rsi
  __int64 v7; // rax

  result = sub_B91A00(a2);
  if ( (_DWORD)result )
  {
    v4 = result;
    for ( i = 0; i != v4; ++i )
    {
      v6 = i;
      v7 = sub_B91A10(a2, v6);
      result = sub_A46690(a1, 0, v7);
    }
  }
  return result;
}
