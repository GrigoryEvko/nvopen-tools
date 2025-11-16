// Function: sub_2C0DFD0
// Address: 0x2c0dfd0
//
bool __fastcall sub_2C0DFD0(__int64 a1)
{
  _QWORD *v2; // rbx

  if ( *(_BYTE *)(a1 - 32) != 28 )
    return 0;
  v2 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * *(unsigned int *)(a1 + 80));
  return v2 != sub_2C0DDE0(*(_QWORD **)(a1 + 72), (__int64)v2, a1 - 40);
}
