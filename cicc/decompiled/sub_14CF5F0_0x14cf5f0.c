// Function: sub_14CF5F0
// Address: 0x14cf5f0
//
__int64 __fastcall sub_14CF5F0(__int64 a1, char a2)
{
  __int64 v2; // rdi

  v2 = *(_WORD *)(a1 + 18) & 0x7FFF;
  if ( a2 )
    return dword_428FE20[(unsigned int)sub_15FF0F0(v2) - 32];
  else
    return dword_428FE20[(unsigned int)(v2 - 32)];
}
