// Function: sub_284B710
// Address: 0x284b710
//
bool __fastcall sub_284B710(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  bool result; // al

  v2 = *(_QWORD *)(a2 + 24);
  result = 0;
  if ( *(_QWORD *)(v2 + 40) == *a1 )
    return *(_BYTE *)v2 != 84;
  return result;
}
