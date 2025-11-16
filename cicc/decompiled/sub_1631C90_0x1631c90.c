// Function: sub_1631C90
// Address: 0x1631c90
//
void __fastcall sub_1631C90(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax

  *(_QWORD *)(a2 + 40) = 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v2 = *(_QWORD *)(a1 + 80);
    if ( v2 )
    {
      v3 = sub_16498B0(a2);
      sub_164D860(v2, v3);
    }
  }
}
