// Function: sub_15601E0
// Address: 0x15601e0
//
__int64 __fastcall sub_15601E0(_QWORD *a1, int a2)
{
  unsigned int v2; // r12d

  v2 = 0;
  if ( a2 != -1 )
    v2 = a2 + 1;
  if ( *a1 && (unsigned int)sub_15601D0((__int64)a1) > v2 )
    return *(_QWORD *)(*a1 + 8LL * v2 + 32);
  else
    return 0;
}
