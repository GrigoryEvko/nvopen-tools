// Function: sub_2A81D20
// Address: 0x2a81d20
//
__int64 __fastcall sub_2A81D20(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  unsigned int v3; // r12d
  int v4; // eax

  v2 = (_QWORD *)*a1;
  if ( (_QWORD *)*a1 == a1 )
    return 0;
  v3 = 0;
  do
  {
    v4 = (*(__int64 (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v2[2] + 16LL))(v2[2], a2);
    v2 = (_QWORD *)*v2;
    v3 |= v4;
  }
  while ( a1 != v2 );
  return v3;
}
