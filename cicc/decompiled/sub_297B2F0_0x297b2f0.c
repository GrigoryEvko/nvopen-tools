// Function: sub_297B2F0
// Address: 0x297b2f0
//
__int64 __fastcall sub_297B2F0(__int64 a1, char a2)
{
  __int64 result; // rax

  result = (unsigned __int8)qword_5007128;
  *(_QWORD *)(a1 + 8) = 0;
  if ( !a2 )
    a2 = result;
  *(_BYTE *)a1 = a2;
  return result;
}
