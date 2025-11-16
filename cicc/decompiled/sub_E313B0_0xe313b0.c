// Function: sub_E313B0
// Address: 0xe313b0
//
__int64 __fastcall sub_E313B0(int a1)
{
  __int64 result; // rax
  int v2; // edx
  unsigned int v3; // eax

  result = 1;
  if ( (unsigned __int8)(a1 - 48) > 9u )
  {
    v2 = a1 - 97;
    if ( (unsigned __int8)(a1 - 97) > 0x19u )
    {
      v3 = a1 - 65;
      LOBYTE(v3) = (unsigned __int8)(a1 - 65) <= 0x19u;
      LOBYTE(v2) = (_BYTE)a1 == 95;
      return v2 | v3;
    }
  }
  return result;
}
