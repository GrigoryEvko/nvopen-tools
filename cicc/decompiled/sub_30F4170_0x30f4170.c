// Function: sub_30F4170
// Address: 0x30f4170
//
bool __fastcall sub_30F4170(__int64 a1, __int64 a2, __int64 a3)
{
  if ( *(_WORD *)(a2 + 24) == 8 )
    return *(_QWORD *)(a2 + 48) != a3;
  else
    return sub_DADE90(*(_QWORD *)(a1 + 104), a2, a3);
}
