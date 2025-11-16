// Function: sub_85B780
// Address: 0x85b780
//
__int64 __fastcall sub_85B780(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  v1 = a1[11];
  if ( (*(_BYTE *)(v1 + 89) & 0x40) != 0 )
    return *(_QWORD *)(*a1 + 8LL);
  result = (*(_BYTE *)(v1 + 89) & 8) != 0 ? *(_QWORD *)(v1 + 24) : *(_QWORD *)(v1 + 8);
  if ( !result )
    return *(_QWORD *)(*a1 + 8LL);
  return result;
}
