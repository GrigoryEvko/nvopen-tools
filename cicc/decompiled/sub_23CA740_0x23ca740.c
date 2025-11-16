// Function: sub_23CA740
// Address: 0x23ca740
//
__int64 __fastcall sub_23CA740(__int64 a1)
{
  int v1; // ebx

  if ( *(_DWORD *)(a1 + 8) == 1 && (unsigned __int8)sub_23CC720() )
    return 0;
  v1 = (**(__int64 (__fastcall ***)(__int64))a1)(a1);
  return v1 + 1 - (unsigned int)sub_23CC6E0(a1);
}
