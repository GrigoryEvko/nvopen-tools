// Function: sub_1F80610
// Address: 0x1f80610
//
__int64 __fastcall sub_1F80610(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)a1 = v3;
  if ( v3 )
    sub_1623A60(a1, v3, 2);
  result = *(unsigned int *)(a2 + 64);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
