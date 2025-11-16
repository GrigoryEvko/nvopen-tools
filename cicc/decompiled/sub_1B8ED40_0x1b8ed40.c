// Function: sub_1B8ED40
// Address: 0x1b8ed40
//
__int64 __fastcall sub_1B8ED40(__int64 a1)
{
  unsigned __int8 v1; // al
  char v2; // dl
  char v4; // al

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
  {
    if ( v1 == 5 )
    {
      v4 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
      if ( v4 == 16 )
        v4 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
      if ( (unsigned __int8)(v4 - 1) <= 5u || *(_WORD *)(a1 + 18) == 52 )
        goto LABEL_6;
    }
  }
  else
  {
    v2 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    if ( v2 == 16 )
      v2 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 8LL);
    if ( (unsigned __int8)(v2 - 1) <= 5u || v1 == 76 )
LABEL_6:
      sub_15F2440(a1, -1);
  }
  return a1;
}
