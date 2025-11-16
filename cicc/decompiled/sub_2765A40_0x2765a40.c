// Function: sub_2765A40
// Address: 0x2765a40
//
bool __fastcall sub_2765A40(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rdx

  if ( a2 != a1 )
  {
    v3 = *a3;
    do
    {
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 24) + 40LL) == v3 )
        break;
      do
        a1 = *(_QWORD *)(a1 + 8);
      while ( a1 && (unsigned __int8)(**(_BYTE **)(a1 + 24) - 30) > 0xAu );
    }
    while ( a2 != a1 );
  }
  return a2 != a1;
}
