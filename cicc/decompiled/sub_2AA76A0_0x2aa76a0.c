// Function: sub_2AA76A0
// Address: 0x2aa76a0
//
__int64 __fastcall sub_2AA76A0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax

  LOBYTE(v2) = *(_DWORD *)a2 == 1;
  return (*(unsigned __int8 *)(a2 + 4) ^ 1) & v2;
}
