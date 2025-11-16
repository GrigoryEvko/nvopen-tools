// Function: sub_124BDF0
// Address: 0x124bdf0
//
bool __fastcall sub_124BDF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  _QWORD *v9; // rax

  if ( a6 && ((unsigned int)sub_EA1780(a3) || (unsigned int)sub_EA1630(a3) == 10) )
    return 0;
  v9 = *(_QWORD **)a3;
  if ( !*(_QWORD *)a3 )
  {
    if ( (*(_BYTE *)(a3 + 9) & 0x70) != 0x20 || *(char *)(a3 + 8) < 0 )
      BUG();
    *(_BYTE *)(a3 + 8) |= 8u;
    v9 = sub_E807D0(*(_QWORD *)(a3 + 24));
    *(_QWORD *)a3 = v9;
  }
  return v9[1] == *(_QWORD *)(a4 + 8);
}
