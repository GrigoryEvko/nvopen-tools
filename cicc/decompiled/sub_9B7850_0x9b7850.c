// Function: sub_9B7850
// Address: 0x9b7850
//
char __fastcall sub_9B7850(__int64 a1, unsigned int a2, __int64 a3)
{
  char result; // al

  if ( a3 && (unsigned __int8)sub_B60C40(a1) )
    return sub_DFAAA0(a3, (unsigned int)a1, a2);
  result = a2 == 0;
  if ( (_DWORD)a1 == 179 )
    return a2 <= 1;
  return result;
}
