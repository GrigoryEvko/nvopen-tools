// Function: sub_2AA7F60
// Address: 0x2aa7f60
//
__int64 __fastcall sub_2AA7F60(_QWORD **a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rax

  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 <= 0x1Cu )
    return 0;
  **a1 = v2;
  v3 = *(_BYTE **)(a2 - 32);
  if ( *v3 <= 0x1Cu )
    return 0;
  *a1[1] = v3;
  return 1;
}
