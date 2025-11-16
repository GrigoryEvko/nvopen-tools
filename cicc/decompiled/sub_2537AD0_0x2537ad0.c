// Function: sub_2537AD0
// Address: 0x2537ad0
//
__int64 __fastcall sub_2537AD0(_BYTE *a1)
{
  char v1; // al
  __int64 v2; // r8

  v1 = *a1;
  if ( (unsigned __int8)(*a1 - 61) <= 1u )
    return *((_QWORD *)a1 - 4);
  if ( v1 == 65 )
    return *((_QWORD *)a1 - 12);
  v2 = 0;
  if ( v1 == 66 )
    return *((_QWORD *)a1 - 8);
  return v2;
}
