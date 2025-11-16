// Function: sub_15FA630
// Address: 0x15fa630
//
__int64 __fastcall sub_15FA630(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned int v3; // r8d

  v3 = 0;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 || *a2 != *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
    return 0;
  LOBYTE(v3) = *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 11;
  return v3;
}
