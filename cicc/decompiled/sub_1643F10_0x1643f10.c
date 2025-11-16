// Function: sub_1643F10
// Address: 0x1643f10
//
__int64 __fastcall sub_1643F10(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax
  unsigned int v3; // eax

  v1 = *(unsigned __int8 *)(a1 + 8);
  result = 1;
  if ( (_BYTE)v1 != 11 )
  {
    v3 = v1 - 1;
    LOBYTE(v3) = (unsigned __int8)(v1 - 1) <= 5u;
    LOBYTE(v1) = (_BYTE)v1 == 15;
    return v1 | v3;
  }
  return result;
}
