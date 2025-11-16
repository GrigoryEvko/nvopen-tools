// Function: sub_13A6B70
// Address: 0x13a6b70
//
__int64 __fastcall sub_13A6B70(__int64 a1, _QWORD **a2)
{
  _QWORD *v2; // rax
  unsigned int i; // r8d

  v2 = *a2;
  for ( i = 1; v2; ++i )
    v2 = (_QWORD *)*v2;
  return i;
}
