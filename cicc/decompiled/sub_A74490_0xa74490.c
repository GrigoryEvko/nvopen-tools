// Function: sub_A74490
// Address: 0xa74490
//
__int64 __fastcall sub_A74490(_QWORD *a1, int a2)
{
  if ( !*a1 )
    return 0;
  if ( (unsigned int)sub_A74480((__int64)a1) > a2 + 1 )
    return *(_QWORD *)(*a1 + 8LL * (unsigned int)(a2 + 1) + 48);
  return 0;
}
