// Function: sub_14A07C0
// Address: 0x14a07c0
//
__int64 __fastcall sub_14A07C0(__int64 a1, __int64 a2, int a3)
{
  if ( a3 < 0 )
    a3 = *(_DWORD *)(a2 + 12) - 1;
  return (unsigned int)(a3 + 1);
}
