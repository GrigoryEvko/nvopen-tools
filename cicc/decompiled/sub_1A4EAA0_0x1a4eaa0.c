// Function: sub_1A4EAA0
// Address: 0x1a4eaa0
//
__int64 __fastcall sub_1A4EAA0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 == *(_QWORD *)(a1 + 40) )
    return 0;
  *(_QWORD *)(a1 + 32) = v1 + 8;
  return 1;
}
