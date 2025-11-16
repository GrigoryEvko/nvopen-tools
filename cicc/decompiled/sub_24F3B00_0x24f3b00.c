// Function: sub_24F3B00
// Address: 0x24f3b00
//
unsigned __int8 *__fastcall sub_24F3B00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *result; // rax

  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( **(_BYTE **)(a1 - 32 * v2) != 17 )
    sub_C64ED0("size argument to coro.id.async must be constant", 1u);
  if ( **(_BYTE **)(a1 + 32 * (1 - v2)) != 17 )
    sub_C64ED0("alignment argument to coro.id.async must be constant", 1u);
  if ( **(_BYTE **)(a1 + 32 * (2 - v2)) != 17 )
    sub_C64ED0("storage argument offset to coro.id.async must be constant", 1u);
  result = sub_BD3990(*(unsigned __int8 **)(a1 + 32 * (3 - v2)), a2);
  if ( *result != 3 )
    sub_C64ED0("llvm.coro.id.async async function pointer not a global", 1u);
  return result;
}
