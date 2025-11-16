// Function: sub_269A220
// Address: 0x269a220
//
__int64 __fastcall sub_269A220(__int64 a1, __int64 a2)
{
  __int64 i; // r13
  unsigned __int16 v3; // cx
  __int64 v5[7]; // [rsp+8h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v3 = ((*(_WORD *)(i - 54) >> 4) & 0x3FF) - 71;
    if ( v3 <= 0x14u && ((1LL << v3) & 0x100021) != 0 && (unsigned __int8)sub_26747F0(i - 56) )
    {
      v5[0] = i - 56;
      sub_2699F90(a1, v5);
    }
  }
  return a1;
}
