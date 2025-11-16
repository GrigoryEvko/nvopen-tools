// Function: sub_ACABB0
// Address: 0xacabb0
//
__int64 __fastcall sub_ACABB0(__int64 a1, unsigned int a2)
{
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) - 16) > 2u )
    return sub_ACA8A0(*(__int64 ***)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL) + 8LL * a2));
  else
    return sub_ACA8A0(*(__int64 ***)(*(_QWORD *)(a1 + 8) + 24LL));
}
