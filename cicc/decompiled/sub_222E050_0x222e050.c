// Function: sub_222E050
// Address: 0x222e050
//
void __fastcall sub_222E050(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A067E8;
  sub_222DFB0(a1, 0);
  sub_222DFF0(a1);
  v2 = *(_QWORD *)(a1 + 200);
  if ( v2 != a1 + 64 )
  {
    if ( v2 )
      j_j___libc_free_0_0(v2);
    *(_QWORD *)(a1 + 200) = 0;
  }
  sub_2209150((volatile signed __int32 **)(a1 + 208));
}
