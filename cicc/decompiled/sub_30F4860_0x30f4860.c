// Function: sub_30F4860
// Address: 0x30f4860
//
bool __fastcall sub_30F4860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 v8; // r13

  if ( *(_WORD *)(a2 + 24) != 8 )
    return 0;
  if ( *(_QWORD *)(a2 + 40) == 2 )
  {
    v7 = **(_QWORD **)(a2 + 32);
    v8 = sub_D33D80((_QWORD *)a2, *(_QWORD *)(a1 + 104), a3, a4, a5);
    if ( sub_DADE90(*(_QWORD *)(a1 + 104), v7, a3) )
      return sub_DADE90(*(_QWORD *)(a1 + 104), v8, a3);
  }
  return 0;
}
