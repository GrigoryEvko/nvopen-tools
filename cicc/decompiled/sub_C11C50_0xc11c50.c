// Function: sub_C11C50
// Address: 0xc11c50
//
__int64 __fastcall sub_C11C50(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(a1 + 48) == v1 || !v1 )
    return 0;
  else
    return v1 - 56;
}
