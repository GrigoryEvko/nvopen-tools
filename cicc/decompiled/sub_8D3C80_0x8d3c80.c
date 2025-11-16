// Function: sub_8D3C80
// Address: 0x8d3c80
//
_BOOL8 __fastcall sub_8D3C80(__int64 a1, const char *a2)
{
  char v2; // al
  _QWORD *v3; // rdx

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 140);
    if ( v2 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( (unsigned __int8)(v2 - 9) <= 2u
    && qword_4D049B8
    && (v3 = *(_QWORD **)a1, (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x10) == 0)
    && v3[8] == qword_4D049B8[11] )
  {
    return strcmp(*(const char **)(*v3 + 8LL), a2) == 0;
  }
  else
  {
    return 0;
  }
}
