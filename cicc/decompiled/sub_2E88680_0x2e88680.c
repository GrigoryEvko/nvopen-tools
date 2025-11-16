// Function: sub_2E88680
// Address: 0x2e88680
//
unsigned __int64 __fastcall sub_2E88680(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  signed __int64 v5; // rcx
  _BYTE *v6; // rsi
  unsigned __int8 v7; // r11
  unsigned __int8 v8; // r9
  unsigned __int8 v9; // r12
  unsigned __int8 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // r12
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // r8
  _QWORD *v15; // r11

  result = *(_QWORD *)(a1 + 48);
  v5 = result & 0xFFFFFFFFFFFFFFF8LL;
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    if ( !a3 )
      return result;
    v15 = 0;
    LODWORD(result) = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, v11, result, a3);
  }
  result &= 7u;
  if ( (_DWORD)result != 3 )
  {
    if ( !a3 )
      return result;
    v13 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    v14 = 0;
    if ( (_DWORD)result != 2 )
    {
      if ( (_DWORD)result == 1 )
      {
        v14 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        v13 = 0;
      }
      else
      {
        if ( !(_DWORD)result )
        {
          *(_QWORD *)(a1 + 48) = v5;
          v15 = (_QWORD *)(a1 + 48);
          v5 = 1;
          v11 = 0;
          v12 = 0;
          v13 = 0;
          return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, v11, result, a3);
        }
        v13 = 0;
      }
    }
    v5 = 0;
    v15 = 0;
    LODWORD(result) = 0;
    v11 = 0;
    v12 = 0;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, v11, result, a3);
  }
  result = 0;
  v6 = (_BYTE *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_BYTE *)(v5 + 9) )
    result = *(_QWORD *)(v5
                       + 8LL * (*(unsigned __int8 *)(v5 + 7) + *(unsigned __int8 *)(v5 + 6))
                       + 8 * (*(int *)v5 + (__int64)(*(unsigned __int8 *)(v5 + 5) + *(unsigned __int8 *)(v5 + 4)))
                       + 16);
  if ( a3 != result )
  {
    LODWORD(result) = 0;
    v5 = *(int *)v5;
    v7 = v6[4];
    v8 = v6[5];
    v9 = v6[6];
    v10 = v6[7];
    if ( v6[8] )
      LODWORD(result) = *(_DWORD *)&v6[8 * v5 + 16 + 8 * v8 + 8 * v7 + 8 * v10 + 8 * v9];
    v11 = 0;
    if ( v10 )
      v11 = *(_QWORD *)&v6[8 * v9 + 16 + 8 * v5 + 8 * v8 + 8 * v7];
    if ( v9 )
      v12 = *(_QWORD *)&v6[8 * v5 + 16 + 8 * v8 + 8 * v7];
    else
      v12 = 0;
    if ( v8 )
      v13 = *(_QWORD *)&v6[8 * v5 + 16 + 8 * v7];
    else
      v13 = 0;
    v14 = 0;
    if ( v7 )
      v14 = *(_QWORD *)&v6[8 * v5 + 16];
    v15 = v6 + 16;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, v11, result, a3);
  }
  return result;
}
