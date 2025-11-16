// Function: sub_6E9FE0
// Address: 0x6e9fe0
//
void __fastcall sub_6E9FE0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  bool v3; // zf

  sub_6E2E50(5, (__int64)a2);
  a2[18] = a1;
  *a2 = sub_72CBA0();
  *(_QWORD *)((char *)a2 + 68) = *(_QWORD *)sub_6E1A20(a1);
  *(_QWORD *)((char *)a2 + 76) = *(_QWORD *)sub_6E1A60(a1);
  v2 = *(_BYTE *)(a1 + 9);
  if ( (v2 & 0x20) == 0 )
  {
    v3 = *(_BYTE *)(a1 + 8) == 1;
    *(_BYTE *)(a1 + 9) = v2 | 0x20;
    if ( v3 )
      sub_6E9F70(*(_QWORD *)(a1 + 24));
  }
}
