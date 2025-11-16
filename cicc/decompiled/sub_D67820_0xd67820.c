// Function: sub_D67820
// Address: 0xd67820
//
__int64 __fastcall sub_D67820(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 == *(_QWORD *)(a1 + 24) )
    return 0;
  else
    return v1 - 8;
}
