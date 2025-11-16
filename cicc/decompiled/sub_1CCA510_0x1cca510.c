// Function: sub_1CCA510
// Address: 0x1cca510
//
__int64 __fastcall sub_1CCA510(__int64 **a1, __int64 a2)
{
  __int64 *v3; // r12
  unsigned __int8 *v4; // rax
  size_t v5; // rdx
  unsigned int v6; // r8d

  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 4 <= 1 )
    return 1;
  v3 = *a1;
  v4 = (unsigned __int8 *)sub_1649960(a2);
  LOBYTE(v6) = (unsigned int)sub_16D1B30(v3, v4, v5) != -1;
  return v6;
}
