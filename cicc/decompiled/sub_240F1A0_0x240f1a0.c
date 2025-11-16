// Function: sub_240F1A0
// Address: 0x240f1a0
//
__int64 __fastcall sub_240F1A0(__int64 a1, __int64 a2)
{
  __int64 **v2; // rax

  if ( (unsigned __int8)(*(_BYTE *)(a2 + 8) - 15) > 1u )
    return *(_QWORD *)(a1 + 72);
  v2 = (__int64 **)sub_240F000(a1, a2);
  return sub_AC9350(v2);
}
