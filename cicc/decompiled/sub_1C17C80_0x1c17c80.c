// Function: sub_1C17C80
// Address: 0x1c17c80
//
__int64 __fastcall sub_1C17C80(__int64 a1, unsigned __int64 a2)
{
  if ( a1 && a2 > 0x17 && *(_DWORD *)a1 == 2135835629 )
    return *(unsigned __int16 *)(a1 + 10);
  else
    return 0;
}
