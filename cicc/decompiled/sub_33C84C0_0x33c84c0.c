// Function: sub_33C84C0
// Address: 0x33c84c0
//
__int64 __fastcall sub_33C84C0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40LL * a2) + 96LL);
  result = *(_QWORD *)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
