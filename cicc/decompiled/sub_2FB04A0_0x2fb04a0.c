// Function: sub_2FB04A0
// Address: 0x2fb04a0
//
__int64 __fastcall sub_2FB04A0(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // edx
  unsigned int v3; // eax
  bool v4; // cf
  __int64 result; // rax

  v2 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
  v3 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
  v4 = v2 < v3;
  result = v2 > v3;
  if ( v4 )
    return 0xFFFFFFFFLL;
  return result;
}
