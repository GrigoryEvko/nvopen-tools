// Function: sub_AA6050
// Address: 0xaa6050
//
__int64 __fastcall sub_AA6050(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rsi
  int v3; // ecx

  result = *(_QWORD *)(a1 + 56);
  v2 = a1 + 48;
  if ( a1 + 48 != result )
  {
    if ( !result )
      goto LABEL_7;
    *(_DWORD *)(result + 32) = 0;
    result = *(_QWORD *)(result + 8);
    v3 = 1;
    if ( v2 != result )
    {
      while ( result )
      {
        *(_DWORD *)(result + 32) = v3;
        result = *(_QWORD *)(result + 8);
        ++v3;
        if ( v2 == result )
          goto LABEL_6;
      }
LABEL_7:
      MEMORY[0x38] = 0;
      BUG();
    }
  }
LABEL_6:
  *(_WORD *)(a1 + 2) |= 0x8000u;
  return result;
}
