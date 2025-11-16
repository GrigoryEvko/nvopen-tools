// Function: sub_15A9650
// Address: 0x15a9650
//
__int64 __fastcall sub_15A9650(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 result; // rax

  v2 = sub_15A95F0(a1, a2);
  result = sub_1644900(*(_QWORD *)a2, v2);
  if ( *(_BYTE *)(a2 + 8) == 16 )
    return sub_16463B0(result, *(_QWORD *)(a2 + 32));
  return result;
}
