// Function: sub_13A34B0
// Address: 0x13a34b0
//
__int64 __fastcall sub_13A34B0(__int64 *a1, unsigned int a2)
{
  unsigned __int64 v2; // rax
  __int64 result; // rax
  __int64 v4; // rdi

  v2 = *a1;
  if ( (*a1 & 1) != 0 )
  {
    result = 2 * ((v2 >> 58 << 57) | ~(-1LL << (v2 >> 58)) & (~(-1LL << (v2 >> 58)) & (v2 >> 1) | (1LL << a2))) + 1;
    *a1 = result;
  }
  else
  {
    v4 = *(_QWORD *)v2;
    *(_QWORD *)(v4 + 8LL * (a2 >> 6)) |= 1LL << a2;
    return 1LL << a2;
  }
  return result;
}
