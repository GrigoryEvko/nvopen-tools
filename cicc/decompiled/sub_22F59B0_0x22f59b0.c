// Function: sub_22F59B0
// Address: 0x22f59b0
//
__int64 __fastcall sub_22F59B0(__int64 a1, int a2)
{
  __int64 v3; // [rsp+0h] [rbp-10h] BYREF

  if ( a2 )
    sub_22F3E40(&v3, *(_QWORD *)(a1 + 32) + 80LL * (unsigned int)(a2 - 1), a1);
  else
    sub_22F3E40(&v3, 0, 0);
  return v3;
}
