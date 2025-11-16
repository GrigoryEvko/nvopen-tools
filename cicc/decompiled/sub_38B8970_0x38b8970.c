// Function: sub_38B8970
// Address: 0x38b8970
//
void __fastcall sub_38B8970(__int64 a1)
{
  if ( *(_BYTE *)(a1 + 304) == 1 && *(_QWORD *)(a1 + 296) && sub_38B8910(a1) )
  {
    sub_15DE580(
      *(_QWORD *)(a1 + 296),
      (__int64 *)(*(_QWORD *)a1 + 16LL * *(_QWORD *)(a1 + 280)),
      (16LL * *(unsigned int *)(a1 + 8) - 16LL * *(_QWORD *)(a1 + 280)) >> 4);
    *(_QWORD *)(a1 + 280) = *(unsigned int *)(a1 + 8);
  }
}
