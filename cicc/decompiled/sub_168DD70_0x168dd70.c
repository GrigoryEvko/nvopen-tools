// Function: sub_168DD70
// Address: 0x168dd70
//
__int64 __fastcall sub_168DD70(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
