// Function: sub_D949C0
// Address: 0xd949c0
//
__int64 __fastcall sub_D949C0(__int64 a1)
{
  __int64 result; // rax
  unsigned int v3; // edx

  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
    return sub_C44590(a1);
  _RCX = *(_QWORD *)a1;
  v3 = 64;
  __asm { tzcnt   rsi, rcx }
  if ( *(_QWORD *)a1 )
    v3 = _RSI;
  if ( (unsigned int)result > v3 )
    return v3;
  return result;
}
