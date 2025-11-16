// Function: sub_2FF5720
// Address: 0x2ff5720
//
__int64 __fastcall sub_2FF5720(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // r8
  unsigned __int16 *v7; // rsi
  __int64 v8; // rcx
  int v9; // edx
  unsigned __int16 *v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // rcx
  unsigned __int16 *v14; // rax
  __int64 result; // rax
  unsigned int v16; // esi

  v6 = *(_QWORD *)(a1 + 280);
  v7 = *(unsigned __int16 **)(a3 + 16);
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *v7;
  v10 = v7 + 1;
  v11 = (*(_QWORD *)(a1 + 288) - v6) >> 3;
  v12 = 4LL * ((unsigned int)(v11 + 31) >> 5);
  v13 = v12 + v8;
  if ( !v9 )
    v10 = 0;
LABEL_6:
  v14 = v10;
  while ( 1 )
  {
    if ( !v14 )
      return 0;
    if ( a4 == v9 )
      break;
    v9 = *v14;
    v13 += v12;
    ++v14;
    v10 = 0;
    if ( !v9 )
      goto LABEL_6;
  }
  result = 0;
  if ( (_DWORD)v11 )
  {
    v16 = 0;
    while ( !(*(_DWORD *)(*(_QWORD *)(a2 + 8) + result) & *(_DWORD *)(v13 + result)) )
    {
      v16 += 32;
      result += 4;
      if ( (unsigned int)v11 <= v16 )
        return 0;
    }
    __asm { tzcnt   edx, edx }
    return *(_QWORD *)(v6 + 8LL * (_EDX + v16));
  }
  return result;
}
