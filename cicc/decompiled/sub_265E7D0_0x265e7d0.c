// Function: sub_265E7D0
// Address: 0x265e7d0
//
__int64 __fastcall sub_265E7D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = a2;
  if ( *(_DWORD *)(a1 + 16) >= *(_DWORD *)(a2 + 16) )
  {
    a2 = a1;
    a1 = v2;
  }
  return sub_265E6F0(a1, a2);
}
