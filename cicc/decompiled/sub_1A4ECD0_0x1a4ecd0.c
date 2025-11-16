// Function: sub_1A4ECD0
// Address: 0x1a4ecd0
//
__int64 __fastcall sub_1A4ECD0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 == *(_QWORD *)(a1 + 24) )
    return 0;
  *(_QWORD *)(a1 + 16) = v1 + 8;
  return 1;
}
