// Function: sub_168DCB0
// Address: 0x168dcb0
//
__int64 __fastcall sub_168DCB0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) == v1 )
    return 0;
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(v1 + 8);
  return 1;
}
