// Function: sub_133DED0
// Address: 0x133ded0
//
__int64 __fastcall sub_133DED0(_DWORD *a1, unsigned __int64 a2)
{
  __int64 result; // rax

  result = (unsigned int)(0x100000000LL / a2) - ((0x100000000LL % a2 == 0) - 1);
  *a1 = result;
  return result;
}
