// Function: sub_15E4C60
// Address: 0x15e4c60
//
__int64 __fastcall sub_15E4C60(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl

  if ( *(_BYTE *)(a1 + 16) != 1 )
    return (unsigned int)(1 << (*(_DWORD *)(a1 + 32) >> 15)) >> 1;
  v1 = sub_164A820(*(_QWORD *)(a1 - 24));
  v2 = *(_BYTE *)(v1 + 16);
  if ( v2 && v2 != 3 )
    return 0;
  else
    return (unsigned int)(1 << (*(_DWORD *)(v1 + 32) >> 15)) >> 1;
}
