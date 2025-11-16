// Function: sub_130B210
// Address: 0x130b210
//
unsigned __int64 __fastcall sub_130B210(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax

  result = *a1 / a2;
  *a1 = result;
  return result;
}
