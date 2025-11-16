// Function: sub_15A08E0
// Address: 0x15a08e0
//
__int64 __fastcall sub_15A08E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  if ( ((*(_BYTE *)(*(_QWORD *)a1 + 8LL) - 14) & 0xFD) != 0 )
    return sub_15A06D0(
             *(__int64 ***)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) + 8LL * (unsigned int)a2),
             (unsigned int)a2,
             a3,
             a4);
  else
    return sub_15A06D0(**(__int64 ****)(*(_QWORD *)a1 + 16LL), a2, a3, a4);
}
