// Function: sub_88FAD0
// Address: 0x88fad0
//
__int64 __fastcall sub_88FAD0(char a1, __int64 a2, __int64 a3, unsigned int a4)
{
  if ( a1 == 3 )
    return sub_8DCBF0(a3, a2, a4, 0);
  if ( a1 == 2 )
    return sub_8DCCD0(a3, a2, a4, 0);
  return sub_8DCC70(a3, *(_QWORD *)(a2 + 104), a4, 0);
}
