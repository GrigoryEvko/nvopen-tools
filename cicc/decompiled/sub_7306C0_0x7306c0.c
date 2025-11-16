// Function: sub_7306C0
// Address: 0x7306c0
//
_BOOL8 __fastcall sub_7306C0(__int64 a1)
{
  __int64 v1; // rbx
  char v3; // al

  v1 = a1;
  if ( (unsigned int)sub_8D2930(a1)
    || (unsigned int)sub_8D2660(a1)
    || dword_4F077BC && (unsigned int)sub_8D2A90(a1)
    || (unsigned int)sub_8D3D40(a1) )
  {
    return 0;
  }
  while ( 1 )
  {
    v3 = *(_BYTE *)(v1 + 140);
    if ( v3 != 12 )
      break;
    v1 = *(_QWORD *)(v1 + 160);
  }
  return v3 != 0;
}
