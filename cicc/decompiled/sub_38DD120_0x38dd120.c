// Function: sub_38DD120
// Address: 0x38dd120
//
bool __fastcall sub_38DD120(__int64 a1)
{
  __int64 v1; // rdx
  bool result; // al

  v1 = *(_QWORD *)(a1 + 32);
  result = 0;
  if ( *(_QWORD *)(a1 + 24) != v1 )
    return *(_QWORD *)(v1 - 72) == 0;
  return result;
}
