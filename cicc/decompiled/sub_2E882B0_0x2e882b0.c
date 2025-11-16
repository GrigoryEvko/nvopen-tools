// Function: sub_2E882B0
// Address: 0x2e882b0
//
unsigned __int64 __fastcall sub_2E882B0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // rcx
  unsigned __int8 v6; // r8
  _BYTE *v7; // rsi
  __int64 v8; // rbx
  unsigned __int8 v9; // r11
  unsigned __int8 v10; // r9
  unsigned __int8 v11; // r12
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
    v8 = 0;
    LODWORD(result) = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, a3, result, v8);
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
          v8 = 0;
          v12 = 0;
          v13 = 0;
          return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, a3, result, v8);
        }
        v13 = 0;
      }
    }
    v5 = 0;
    v15 = 0;
    v8 = 0;
    LODWORD(result) = 0;
    v12 = 0;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, a3, result, v8);
  }
  v6 = *(_BYTE *)(v5 + 7);
  v7 = (_BYTE *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  result = 0;
  if ( v6 )
    result = *(_QWORD *)(v5
                       + 8LL * *(unsigned __int8 *)(v5 + 6)
                       + 8 * (*(int *)v5 + (__int64)(*(unsigned __int8 *)(v5 + 5) + *(unsigned __int8 *)(v5 + 4)))
                       + 16);
  if ( a3 != result )
  {
    v8 = 0;
    v5 = *(int *)v5;
    v9 = v7[4];
    v10 = v7[5];
    v11 = v7[6];
    if ( v7[9] )
      v8 = *(_QWORD *)&v7[8 * v6 + 16 + 8 * v11 + 8 * v5 + 8 * v10 + 8 * v9];
    LODWORD(result) = 0;
    if ( v7[8] )
      LODWORD(result) = *(_DWORD *)&v7[8 * v5 + 16 + 8 * v10 + 8 * v9 + 8 * v6 + 8 * v11];
    if ( v11 )
      v12 = *(_QWORD *)&v7[8 * v5 + 16 + 8 * v10 + 8 * v9];
    else
      v12 = 0;
    if ( v10 )
      v13 = *(_QWORD *)&v7[8 * v5 + 16 + 8 * v9];
    else
      v13 = 0;
    v14 = 0;
    if ( v9 )
      v14 = *(_QWORD *)&v7[8 * v5 + 16];
    v15 = v7 + 16;
    return sub_2E867B0(a1, a2, v15, v5, v14, v13, v12, a3, result, v8);
  }
  return result;
}
