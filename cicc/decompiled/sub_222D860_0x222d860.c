// Function: sub_222D860
// Address: 0x222d860
//
unsigned __int64 __fastcall sub_222D860(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v5; // rax
  char *v6; // rdi
  __int64 v7; // rdx
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rax

  if ( a2 > 0xD )
  {
    v5 = 297;
    v6 = (char *)&unk_43619D0;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = (unsigned __int64 *)&v6[8 * (v5 >> 1)];
        if ( *v8 >= a2 )
          break;
        v6 = (char *)(v8 + 1);
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          goto LABEL_9;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
LABEL_9:
    v9 = -1;
    if ( v6 != (char *)&unk_4362318 )
    {
      _FST7 = (long double)*(unsigned __int64 *)v6 * *(float *)a1;
      __asm { frndint }
      if ( _FST7 < 9.223372e18 )
      {
        *(_QWORD *)(a1 + 8) = (__int64)_FST7;
        return *(_QWORD *)v6;
      }
      v9 = (__int64)(_FST7 - 9.223372e18) ^ 0x8000000000000000LL;
    }
    *(_QWORD *)(a1 + 8) = v9;
    return *(_QWORD *)v6;
  }
  result = 1;
  if ( a2 )
  {
    result = byte_4361990[a2];
    _FST7 = (long double)byte_4361990[a2] * *(float *)a1;
    __asm { frndint }
    if ( _FST7 < 9.223372e18 )
    {
      *(_QWORD *)(a1 + 8) = (__int64)_FST7;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = (__int64)(_FST7 - 9.223372e18);
      *(_QWORD *)(a1 + 8) ^= 0x8000000000000000LL;
    }
  }
  return result;
}
