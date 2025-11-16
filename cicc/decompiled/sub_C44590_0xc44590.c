// Function: sub_C44590
// Address: 0xc44590
//
__int64 __fastcall sub_C44590(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdx
  __int64 *v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // edx
  __int64 result; // rax

  v2 = *(unsigned int *)(a1 + 8);
  v3 = (unsigned __int64)(v2 + 63) >> 6;
  if ( !v3 )
    return 0;
  v4 = *(__int64 **)a1;
  v5 = *(_QWORD *)a1 + 8LL * (unsigned int)(v3 - 1) + 8;
  v6 = 0;
  while ( 1 )
  {
    _RCX = *v4;
    if ( *v4 )
      break;
    ++v4;
    v6 += 64;
    if ( v4 == (__int64 *)v5 )
      goto LABEL_6;
  }
  __asm { tzcnt   rcx, rcx }
  v6 += _RCX;
LABEL_6:
  result = (unsigned int)v2;
  if ( v6 <= (unsigned int)v2 )
    return v6;
  return result;
}
