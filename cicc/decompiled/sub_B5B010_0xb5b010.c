// Function: sub_B5B010
// Address: 0xb5b010
//
char __fastcall sub_B5B010(int a1, __int64 a2, __int64 a3, int a4)
{
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax

  v4 = sub_B5A790(a1, a2, a3, a4);
  v5 = HIDWORD(v4);
  if ( BYTE4(v4) )
    LOBYTE(v5) = (unsigned int)(v4 - 38) <= 0xC;
  return v5;
}
