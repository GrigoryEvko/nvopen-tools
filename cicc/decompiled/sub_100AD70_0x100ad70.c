// Function: sub_100AD70
// Address: 0x100ad70
//
bool __fastcall sub_100AD70(_QWORD *a1, int a2, unsigned __int8 *a3)
{
  bool result; // al

  result = 0;
  if ( a2 + 29 == *a3 && *a1 == *((_QWORD *)a3 - 8) )
    return *((_QWORD *)a3 - 4) == a1[1];
  return result;
}
