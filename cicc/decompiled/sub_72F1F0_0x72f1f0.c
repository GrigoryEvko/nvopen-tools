// Function: sub_72F1F0
// Address: 0x72f1f0
//
__int64 *__fastcall sub_72F1F0(__int64 a1)
{
  __int64 *result; // rax

  result = *(__int64 **)(a1 + 192);
  if ( !result && (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
    return sub_72DB50(a1, 2);
  return result;
}
