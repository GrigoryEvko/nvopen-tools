// Function: sub_22DADF0
// Address: 0x22dadf0
//
__int64 __fastcall sub_22DADF0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int i; // r8d

  v1 = *(_QWORD *)(a1 + 8);
  for ( i = 0; v1; ++i )
    v1 = *(_QWORD *)(v1 + 8);
  return i;
}
