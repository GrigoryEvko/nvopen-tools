// Function: sub_2FF6970
// Address: 0x2ff6970
//
__int64 __fastcall sub_2FF6970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rax
  bool v8; // zf
  unsigned int v9; // r8d
  __int64 result; // rax
  unsigned int v11; // ecx

  if ( a2 == a3 )
    return a2;
  if ( !a2 || !a3 )
    return 0;
  v4 = *(_QWORD *)(a1 + 280);
  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_QWORD *)(a3 + 8);
  v7 = (*(_QWORD *)(a1 + 288) - v4) >> 3;
  v8 = (_DWORD)v7 == 0;
  v9 = v7;
  result = 0;
  if ( !v8 )
  {
    v11 = 0;
    while ( !(*(_DWORD *)(v6 + result) & *(_DWORD *)(v5 + result)) )
    {
      v11 += 32;
      result += 4;
      if ( v9 <= v11 )
        return 0;
    }
    __asm { tzcnt   edx, edx }
    return *(_QWORD *)(v4 + 8LL * (_EDX + v11));
  }
  return result;
}
