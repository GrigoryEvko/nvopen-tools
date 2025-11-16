// Function: sub_25AC5C0
// Address: 0x25ac5c0
//
__int64 __fastcall sub_25AC5C0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
