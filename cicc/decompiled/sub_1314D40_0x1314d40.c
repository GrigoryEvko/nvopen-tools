// Function: sub_1314D40
// Address: 0x1314d40
//
int __fastcall sub_1314D40(__int64 a1, __int64 a2)
{
  int result; // eax

  if ( !*(_QWORD *)(a2 + 69440) )
    sub_1314B60(a1, a2, a2 + 69320, *(volatile signed __int64 **)(a2 + 72896), a2 + 10728, 0, 1u);
  result = byte_5260DD0[0];
  if ( byte_5260DD0[0] )
  {
    result = *(unsigned __int8 *)(unk_5260DD8
                                + 208 * ((unsigned __int64)*(unsigned int *)(a2 + 78928) % unk_5260D48)
                                + 172);
    if ( (_BYTE)result )
      return sub_1314970(a1, a2, a2 + 69320, 0);
  }
  return result;
}
