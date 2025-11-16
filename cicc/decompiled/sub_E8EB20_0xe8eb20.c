// Function: sub_E8EB20
// Address: 0xe8eb20
//
bool __fastcall sub_E8EB20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rax

  if ( *(_QWORD *)a3 )
    return *(_QWORD *)(a4 + 8) == *(_QWORD *)(*(_QWORD *)a3 + 8LL);
  if ( (*(_BYTE *)(a3 + 9) & 0x70) != 0x20 || *(char *)(a3 + 8) < 0 )
    BUG();
  *(_BYTE *)(a3 + 8) |= 8u;
  v5 = sub_E807D0(*(_QWORD *)(a3 + 24));
  *(_QWORD *)a3 = v5;
  return *(_QWORD *)(a4 + 8) == v5[1];
}
