// Function: sub_141FFD0
// Address: 0x141ffd0
//
void __fastcall sub_141FFD0(__int64 a1)
{
  __int64 v1; // rax

  if ( a1 )
  {
    v1 = *(_QWORD *)(a1 + 112);
    if ( v1 != -8 && v1 != 0 && v1 != -16 )
      sub_1649B30(a1 + 96);
    sub_164BE60(a1);
    sub_1648B90(a1);
  }
}
