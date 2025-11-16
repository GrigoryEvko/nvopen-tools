// Function: sub_3258F50
// Address: 0x3258f50
//
unsigned __int64 __fastcall sub_3258F50(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx

  v2 = *(_QWORD *)(a1 + 8);
  if ( a2 )
    return sub_E808D0(a2, *(_BYTE *)(a1 + 27) != 0 ? 0x72 : 0, *(_QWORD **)(v2 + 216), 0);
  else
    return sub_E81A90(0, *(_QWORD **)(v2 + 216), 0, 0);
}
