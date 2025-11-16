// Function: sub_16497E0
// Address: 0x16497e0
//
__int64 __fastcall sub_16497E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  char v5; // r9
  unsigned int v6; // r8d
  __int64 v7; // r10
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rax

  result = a1;
  if ( *(_BYTE *)(a1 + 16) == 77 && *(_QWORD *)(a1 + 40) == a2 )
  {
    v4 = 0x17FFFFFFE8LL;
    v5 = *(_BYTE *)(a1 + 23) & 0x40;
    v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    if ( v6 )
    {
      v7 = a1 - 24LL * v6;
      v8 = 24LL * *(unsigned int *)(a1 + 56) + 8;
      v9 = 0;
      do
      {
        v10 = v7;
        if ( v5 )
          v10 = *(_QWORD *)(result - 8);
        if ( a3 == *(_QWORD *)(v10 + v8) )
        {
          v4 = 24 * v9;
          goto LABEL_11;
        }
        ++v9;
        v8 += 8;
      }
      while ( v6 != (_DWORD)v9 );
      v4 = 0x17FFFFFFE8LL;
    }
LABEL_11:
    if ( v5 )
      v11 = *(_QWORD *)(result - 8);
    else
      v11 = result - 24LL * v6;
    return *(_QWORD *)(v11 + v4);
  }
  return result;
}
