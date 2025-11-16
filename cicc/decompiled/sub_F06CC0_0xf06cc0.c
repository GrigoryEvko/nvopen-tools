// Function: sub_F06CC0
// Address: 0xf06cc0
//
__int64 __fastcall sub_F06CC0(unsigned __int8 **a1, __int64 a2)
{
  unsigned __int8 *v2; // rcx
  __int64 result; // rax
  unsigned __int8 v4; // dl
  int v5; // esi

  v2 = a1[1];
  result = **a1;
  v4 = **a1;
  if ( !(_BYTE)result )
  {
    v4 = *v2;
    v5 = **(unsigned __int8 **)(a2 + 24);
    if ( (unsigned __int8)v5 > 0x1Cu )
    {
      LOBYTE(result) = (_BYTE)v5 == 76;
      LOBYTE(v5) = (_BYTE)v5 == 82;
      result = v5 | (unsigned int)result;
      v4 |= result;
    }
  }
  *v2 = v4;
  return result;
}
