// Function: sub_15FA9D0
// Address: 0x15fa9d0
//
__int64 __fastcall sub_15FA9D0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdx

  if ( (unsigned int)*(unsigned __int8 *)(a1 + 16) - 11 <= 1 )
    return sub_1595A50(a1, a2);
  v3 = sub_15A0A60(a1, a2);
  v4 = v3;
  if ( *(_BYTE *)(v3 + 16) == 9 )
    return 0xFFFFFFFFLL;
  result = *(_QWORD *)(v3 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
