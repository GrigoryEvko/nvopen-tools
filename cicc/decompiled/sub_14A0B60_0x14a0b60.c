// Function: sub_14A0B60
// Address: 0x14a0b60
//
__int64 __fastcall sub_14A0B60(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rdx
  __int64 result; // rax

  v5 = *(_QWORD *)(a5 + 32);
  result = a2;
  if ( a2 > v5 )
  {
    do
    {
      result = (unsigned int)v5;
      if ( !(_DWORD)v5 )
        break;
      LODWORD(v5) = v5 - 1;
    }
    while ( ((unsigned int)v5 & (unsigned int)result) != 0 );
  }
  return result;
}
