// Function: sub_AA64B0
// Address: 0xaa64b0
//
__int64 __fastcall sub_AA64B0(__int64 a1, __int64 a2)
{
  int v2; // eax

  if ( *(_QWORD *)(a1 + 72) != a2 )
  {
    v2 = -1;
    if ( a2 )
    {
      v2 = *(_DWORD *)(a2 + 88);
      *(_DWORD *)(a2 + 88) = v2 + 1;
    }
    *(_DWORD *)(a1 + 44) = v2;
  }
  return sub_AA63D0((_QWORD *)(a1 + 48), (_QWORD *)(a1 + 72), a2);
}
