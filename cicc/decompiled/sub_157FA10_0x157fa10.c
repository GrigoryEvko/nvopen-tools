// Function: sub_157FA10
// Address: 0x157fa10
//
__int64 __fastcall sub_157FA10(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 *v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 56);
  sub_157F970(a1, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
  {
    v2 = *(_QWORD *)(v1 + 104);
    if ( v2 )
    {
      v3 = sub_16498B0(a1);
      sub_164D860(v2, v3);
    }
  }
  v4 = *(__int64 **)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  result = v5 | *v4 & 7;
  *v4 = result;
  *(_QWORD *)(v5 + 8) = v4;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 24) &= 7uLL;
  return result;
}
