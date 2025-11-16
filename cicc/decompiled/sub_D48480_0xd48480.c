// Function: sub_D48480
// Address: 0xd48480
//
__int64 __fastcall sub_D48480(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 1;
  else
    return (unsigned int)sub_B19060(a1 + 56, *(_QWORD *)(a2 + 40), a3, a4) ^ 1;
}
