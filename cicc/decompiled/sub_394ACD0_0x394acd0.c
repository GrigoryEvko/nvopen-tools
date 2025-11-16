// Function: sub_394ACD0
// Address: 0x394acd0
//
__int64 __fastcall sub_394ACD0(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax

  if ( (unsigned __int8)(a3 - 3) > 7u )
    return *(_QWORD *)(a1 + 40);
  result = *(_QWORD *)(a1 + 56);
  if ( !result )
    return *(_QWORD *)(a1 + 40);
  return result;
}
