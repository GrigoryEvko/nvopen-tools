// Function: sub_1948E70
// Address: 0x1948e70
//
__int64 __fastcall sub_1948E70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 v8; // rax

  v5 = 0;
  result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v7 = 8 * result;
  if ( (_DWORD)result )
  {
    while ( 1 )
    {
      v8 = v5 + 24LL * *(unsigned int *)(a1 + 56) + 8;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        result = *(_QWORD *)(a1 - 8) + v8;
        if ( a2 != *(_QWORD *)result )
          goto LABEL_4;
LABEL_7:
        v5 += 8;
        *(_QWORD *)result = a3;
        if ( v5 == v7 )
          return result;
      }
      else
      {
        result = a1 + v8 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        if ( a2 == *(_QWORD *)result )
          goto LABEL_7;
LABEL_4:
        v5 += 8;
        if ( v5 == v7 )
          return result;
      }
    }
  }
  return result;
}
