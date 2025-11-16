// Function: sub_7D2A80
// Address: 0x7d2a80
//
__int64 __fastcall sub_7D2A80(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 72);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 80) - 4) > 1u )
    return *(_QWORD *)(a1 + 72);
  if ( (unsigned int)sub_85FB30(*(_QWORD *)(a1 + 88)) )
    return a1;
  return v1;
}
