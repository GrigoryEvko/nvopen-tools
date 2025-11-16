// Function: sub_AE43F0
// Address: 0xae43f0
//
__int64 __fastcall sub_AE43F0(__int64 a1, __int64 a2)
{
  if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
    a2 = **(_QWORD **)(a2 + 16);
  return (unsigned int)sub_AE2980(a1, *(_DWORD *)(a2 + 8) >> 8)[3];
}
