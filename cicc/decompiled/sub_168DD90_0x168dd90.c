// Function: sub_168DD90
// Address: 0x168dd90
//
__int64 __fastcall sub_168DD90(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 40) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
