// Function: sub_100ADA0
// Address: 0x100ada0
//
bool __fastcall sub_100ADA0(_QWORD *a1, int a2, unsigned __int8 *a3)
{
  bool result; // al

  result = 0;
  if ( a2 + 29 == *a3 && *a1 == *((_QWORD *)a3 - 8) )
    return *((_QWORD *)a3 - 4) == a1[1];
  return result;
}
