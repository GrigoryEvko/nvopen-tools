// Function: sub_AC2FF0
// Address: 0xac2ff0
//
__int64 __fastcall sub_AC2FF0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 result; // rax

  sub_BD35F0(a1, a2, 17);
  *(_DWORD *)(a1 + 4) &= 0x38000000u;
  v4 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a1 + 32) = v4;
  if ( v4 > 0x40 )
    return sub_C43780(a1 + 24, a3);
  result = *(_QWORD *)a3;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)a3;
  return result;
}
