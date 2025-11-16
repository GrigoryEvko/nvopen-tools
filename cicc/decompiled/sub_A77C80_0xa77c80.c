// Function: sub_A77C80
// Address: 0xa77c80
//
__int64 __fastcall sub_A77C80(__int64 **a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rsi

  v3 = (unsigned int)a3;
  v4 = HIDWORD(a3);
  v5 = a2 << 32;
  v6 = v3 | (a2 << 32);
  if ( !(_BYTE)v4 )
    v6 = v5;
  return sub_A77C60(a1, v6);
}
