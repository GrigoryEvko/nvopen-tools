// Function: sub_A4F550
// Address: 0xa4f550
//
__int64 __fastcall sub_A4F550(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 24) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
