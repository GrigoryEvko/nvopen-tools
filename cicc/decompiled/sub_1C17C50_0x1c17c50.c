// Function: sub_1C17C50
// Address: 0x1c17c50
//
__int64 __fastcall sub_1C17C50(__int64 a1)
{
  _DWORD *v1; // rdx
  unsigned int v2; // r8d

  v1 = *(_DWORD **)(a1 + 8);
  v2 = 0;
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v1 > 0x17u )
    LOBYTE(v2) = *v1 == 2135835629;
  return v2;
}
