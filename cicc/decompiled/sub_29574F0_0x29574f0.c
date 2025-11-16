// Function: sub_29574F0
// Address: 0x29574f0
//
__int64 __fastcall sub_29574F0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 == *(_QWORD *)(a1 + 24) )
    return 0;
  *(_QWORD *)(a1 + 8) = v1 + 8;
  return 1;
}
