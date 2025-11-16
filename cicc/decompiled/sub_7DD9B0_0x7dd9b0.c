// Function: sub_7DD9B0
// Address: 0x7dd9b0
//
void __fastcall sub_7DD9B0(__int64 a1, _QWORD *a2)
{
  __int64 i; // rbx
  _QWORD *j; // rbx

  sub_7DD8B0(*(__m128i **)(a1 + 104), a2);
  for ( i = *(_QWORD *)(a1 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_7DD9B0(*(_QWORD *)(i + 128));
  }
  for ( j = *(_QWORD **)(a1 + 160); j; j = (_QWORD *)*j )
    sub_7DD9B0(j);
}
