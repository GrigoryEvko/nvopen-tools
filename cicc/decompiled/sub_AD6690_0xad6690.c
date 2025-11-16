// Function: sub_AD6690
// Address: 0xad6690
//
__int64 __fastcall sub_AD6690(__int64 a1, __int64 a2)
{
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) - 16) > 2u )
    return sub_AD6530(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL) + 8LL * (unsigned int)a2), (unsigned int)a2);
  else
    return sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 24LL), a2);
}
