// Function: sub_7AED40
// Address: 0x7aed40
//
unsigned __int64 __fastcall sub_7AED40(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rcx

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    v2 = (result >> 3) - 7993 * (result / 0xF9C8);
    *(_QWORD *)(a1 + 8) = qword_4F08580[v2];
    qword_4F08580[v2] = a1;
    return (unsigned __int64)qword_4F08580;
  }
  return result;
}
