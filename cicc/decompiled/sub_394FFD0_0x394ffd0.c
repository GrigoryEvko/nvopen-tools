// Function: sub_394FFD0
// Address: 0x394ffd0
//
__int64 __fastcall sub_394FFD0(__int64 a1, __int64 a2, __int64 *a3)
{
  bool v4; // r8
  __int64 result; // rax

  v4 = sub_394FE80(a1, a2, 3u, 2u, 0);
  result = 0;
  if ( v4 )
  {
    sub_15E7430(
      a3,
      *(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
      1u,
      *(_QWORD **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
      1u,
      *(__int64 **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
      0,
      0,
      0,
      0,
      0);
    return *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  }
  return result;
}
