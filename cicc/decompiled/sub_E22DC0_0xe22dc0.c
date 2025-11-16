// Function: sub_E22DC0
// Address: 0xe22dc0
//
__int64 __fastcall sub_E22DC0(__int64 a1, _QWORD *a2)
{
  char *v2; // rcx
  char v3; // dl

  if ( *a2 )
  {
    v2 = (char *)a2[1];
    v3 = *v2;
    --*a2;
    a2[1] = v2 + 1;
    if ( (unsigned __int8)(v3 - 65) <= 0x16u )
      return byte_3F7C780[(unsigned __int8)(v3 - 65)];
    else
      return 0;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
}
