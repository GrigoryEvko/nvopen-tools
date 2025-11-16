// Function: sub_130B900
// Address: 0x130b900
//
unsigned __int64 __fastcall sub_130B900(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v4; // rax

  v2 = (*(__int64 (__fastcall **)(__int64, __int64))(a2 + 72))(a1, a2 + 24);
  if ( !v2 || !*(_BYTE *)(a2 + 16) )
    return v2;
  v4 = (*(__int64 (__fastcall **)(__int64, __int64))(a2 + 62432))(a1, a2 + 62384);
  if ( v2 > v4 )
    return v4;
  return v2;
}
