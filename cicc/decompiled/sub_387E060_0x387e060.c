// Function: sub_387E060
// Address: 0x387e060
//
__int64 __fastcall sub_387E060(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 80) & 1) == 0 && (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 && !sub_15E4F60(a1) )
    __asm { jmp     rax }
  return 0;
}
