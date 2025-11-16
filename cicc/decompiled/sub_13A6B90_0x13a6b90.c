// Function: sub_13A6B90
// Address: 0x13a6b90
//
__int64 __fastcall sub_13A6B90(__int64 a1, _QWORD **a2)
{
  _QWORD *v2; // rax
  unsigned int i; // r8d
  unsigned int v4; // eax

  v2 = *a2;
  for ( i = 1; v2; ++i )
    v2 = (_QWORD *)*v2;
  v4 = *(_DWORD *)(a1 + 32);
  if ( v4 < i )
    i += *(_DWORD *)(a1 + 36) - v4;
  return i;
}
