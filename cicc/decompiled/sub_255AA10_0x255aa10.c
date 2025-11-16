// Function: sub_255AA10
// Address: 0x255aa10
//
__int64 __fastcall sub_255AA10(_DWORD *a1, unsigned __int8 a2)
{
  int v2; // eax
  __int64 result; // rax

  v2 = (16 * a2) | (4 * a2) & 0xFFCF | a2 & 0xC3;
  LOBYTE(v2) = (16 * a2) & 0x3F | (4 * a2) & 0xF | a2 & 3;
  result = (a2 << 6) | (unsigned int)v2;
  *a1 = result;
  return result;
}
