// Function: sub_1455870
// Address: 0x1455870
//
__int64 __fastcall sub_1455870(__int64 *a1)
{
  unsigned int v1; // edx
  __int64 result; // rax

  v1 = *((_DWORD *)a1 + 2);
  if ( v1 > 0x40 )
    return sub_16A58A0(a1);
  _RCX = *a1;
  result = 64;
  __asm { tzcnt   rsi, rcx }
  if ( *a1 )
    result = (unsigned int)_RSI;
  if ( (unsigned int)result > v1 )
    return v1;
  return result;
}
