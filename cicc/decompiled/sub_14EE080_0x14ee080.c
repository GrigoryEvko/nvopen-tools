// Function: sub_14EE080
// Address: 0x14ee080
//
__int64 __fastcall sub_14EE080(_BYTE *a1)
{
  char v1; // al
  unsigned int v2; // r8d

  v1 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
  if ( v1 == 16 )
    v1 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
  v2 = 1;
  if ( (unsigned __int8)(v1 - 1) > 5u )
    LOBYTE(v2) = a1[16] == 76;
  return v2;
}
