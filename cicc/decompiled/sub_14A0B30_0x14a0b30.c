// Function: sub_14A0B30
// Address: 0x14a0b30
//
__int64 __fastcall sub_14A0B30(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rcx
  __int64 result; // rax

  v5 = *(_QWORD *)(a5 + 32);
  result = a2;
  if ( a2 > v5 && a2 - v5 != 1 )
  {
    for ( result = (unsigned int)v5;
          (_DWORD)result && (((_DWORD)result - 1) & (unsigned int)result) != 0;
          result = (unsigned int)(result - 1) )
    {
      ;
    }
  }
  return result;
}
