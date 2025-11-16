// Function: sub_BD3050
// Address: 0xbd3050
//
bool __fastcall sub_BD3050(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rdx
  bool result; // al

  v2 = *(_QWORD *)(a2 + 24);
  result = 1;
  if ( *(_BYTE *)v2 > 0x1Cu )
    return *(_QWORD *)(v2 + 40) != *a1;
  return result;
}
