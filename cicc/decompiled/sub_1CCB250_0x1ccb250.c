// Function: sub_1CCB250
// Address: 0x1ccb250
//
__int64 __fastcall sub_1CCB250(__int64 a1)
{
  unsigned int v1; // r8d

  v1 = 0;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8 == 5 )
    LOBYTE(v1) = *(_DWORD *)(**(_QWORD **)(a1 - 24) + 8LL) >> 8 == 0;
  return v1;
}
