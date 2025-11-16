// Function: sub_2E88490
// Address: 0x2e88490
//
unsigned __int64 __fastcall sub_2E88490(__int64 a1, __int64 a2, int a3)
{
  unsigned __int64 result; // rax
  signed __int64 v4; // rcx
  int v5; // r8d
  __int64 v6; // r10
  unsigned __int8 v7; // r11
  unsigned __int8 v8; // r9
  unsigned __int8 v9; // r12
  unsigned __int8 v10; // bl
  __int64 v11; // rbx
  __int64 v12; // r12
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // r8
  _QWORD *v15; // r11

  result = *(_QWORD *)(a1 + 48);
  v4 = result & 0xFFFFFFFFFFFFFFF8LL;
  if ( (result & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    if ( !a3 )
      return result;
    v15 = 0;
    v6 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    return sub_2E867B0(a1, a2, v15, v4, v14, v13, v12, v11, a3, v6);
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
          *(_QWORD *)(a1 + 48) = v4;
          v15 = (_QWORD *)(a1 + 48);
          v4 = 1;
          v6 = 0;
          v11 = 0;
          v12 = 0;
          v13 = 0;
          return sub_2E867B0(a1, a2, v15, v4, v14, v13, v12, v11, a3, v6);
        }
        v13 = 0;
      }
    }
    v4 = 0;
    v15 = 0;
    v6 = 0;
    v11 = 0;
    v12 = 0;
    return sub_2E867B0(a1, a2, v15, v4, v14, v13, v12, v11, a3, v6);
  }
  v5 = 0;
  result = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)(v4 + 8) )
    v5 = *(_DWORD *)(v4
                   + 8
                   * (*(int *)v4
                    + *(unsigned __int8 *)(v4 + 7)
                    + *(unsigned __int8 *)(v4 + 6)
                    + (__int64)(*(unsigned __int8 *)(v4 + 5) + *(unsigned __int8 *)(v4 + 4)))
                   + 16);
  if ( a3 != v5 )
  {
    v6 = 0;
    v4 = *(int *)v4;
    v7 = *(_BYTE *)(result + 4);
    v8 = *(_BYTE *)(result + 5);
    v9 = *(_BYTE *)(result + 6);
    v10 = *(_BYTE *)(result + 7);
    if ( *(_BYTE *)(result + 9) )
      v6 = *(_QWORD *)(result + 8LL * (v10 + v9) + 8 * (v4 + v8 + v7) + 16);
    if ( v10 )
      v11 = *(_QWORD *)(result + 8LL * v9 + 8 * (v4 + v8 + v7) + 16);
    else
      v11 = 0;
    if ( v9 )
      v12 = *(_QWORD *)(result + 8 * (v4 + v8 + v7) + 16);
    else
      v12 = 0;
    if ( v8 )
      v13 = *(_QWORD *)(result + 8 * (v4 + v7) + 16);
    else
      v13 = 0;
    v14 = 0;
    if ( v7 )
      v14 = *(_QWORD *)(result + 8 * v4 + 16);
    v15 = (_QWORD *)(result + 16);
    return sub_2E867B0(a1, a2, v15, v4, v14, v13, v12, v11, a3, v6);
  }
  return result;
}
