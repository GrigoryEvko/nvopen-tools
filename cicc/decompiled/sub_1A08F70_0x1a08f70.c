// Function: sub_1A08F70
// Address: 0x1a08f70
//
__int64 __fastcall sub_1A08F70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  result = sub_19FF700(28, a3, a3, a4);
  if ( !result && *(_DWORD *)(a3 + 8) != 1 )
    return sub_1A086C0(a1, a2, (_QWORD *)a3);
  return result;
}
