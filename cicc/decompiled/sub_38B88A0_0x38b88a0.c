// Function: sub_38B88A0
// Address: 0x38b88a0
//
void __fastcall sub_38B88A0(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 304) == 1 && *(_QWORD *)(a1 + 288) && sub_38B8870(a1) )
  {
    sub_15DC140(
      *(_QWORD *)(a1 + 288),
      (__int64 *)(*(_QWORD *)a1 + 16LL * *(_QWORD *)(a1 + 272)),
      (16LL * *(unsigned int *)(a1 + 8) - 16LL * *(_QWORD *)(a1 + 272)) >> 4);
    *(_QWORD *)(a1 + 272) = *(unsigned int *)(a1 + 8);
  }
}
