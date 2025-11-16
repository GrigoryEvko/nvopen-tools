// Function: sub_2E89130
// Address: 0x2e89130
//
__int64 __fastcall sub_2E89130(__int64 a1)
{
  __int64 v1; // rax

  v1 = 80;
  if ( *(_WORD *)(a1 + 68) != 14 )
    v1 = 0;
  return *(_QWORD *)(a1 + 32) + v1;
}
