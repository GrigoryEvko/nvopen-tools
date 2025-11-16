// Function: sub_2E891A0
// Address: 0x2e891a0
//
__int64 __fastcall sub_2E891A0(__int64 a1)
{
  __int64 v1; // rax

  v1 = 120;
  if ( *(_WORD *)(a1 + 68) != 14 )
    v1 = 40;
  return *(_QWORD *)(a1 + 32) + v1;
}
