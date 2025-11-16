// Function: sub_159A1E0
// Address: 0x159a1e0
//
__int64 __fastcall sub_159A1E0(__int64 a1, unsigned int a2)
{
  if ( ((*(_BYTE *)(*(_QWORD *)a1 + 8LL) - 14) & 0xFD) != 0 )
    return sub_1599EF0(*(__int64 ***)(*(_QWORD *)(*(_QWORD *)a1 + 16LL) + 8LL * a2));
  else
    return sub_1599EF0(**(__int64 ****)(*(_QWORD *)a1 + 16LL));
}
