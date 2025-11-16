// Function: sub_277A9A0
// Address: 0x277a9a0
//
char __fastcall sub_277A9A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v5; // esi
  __int64 v6; // rax
  int v7; // edi
  char result; // al
  int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rsi
  unsigned __int8 *v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r8

  v2 = *(_QWORD *)(a1 - 32);
  if ( !v2
    || *(_BYTE *)v2
    || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a1 + 80)
    || (v5 = *(_DWORD *)(v2 + 36), (v6 = *(_QWORD *)(a2 - 32)) == 0)
    || *(_BYTE *)v6
    || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a2 + 80) )
  {
    BUG();
  }
  v7 = *(_DWORD *)(v6 + 36);
  result = v7 == 228 && v5 == 228;
  if ( !result )
  {
    if ( v7 == 228 && v5 == 230 )
    {
      v9 = *(_DWORD *)(a2 + 4);
      result = sub_2778F20(
                 *(unsigned __int8 **)(a2 + 32 * (2LL - (v9 & 0x7FFFFFF))),
                 *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))));
      if ( result )
        return (unsigned int)**(unsigned __int8 **)(a2 + 32 * (3LL - (v9 & 0x7FFFFFF))) - 12 <= 1;
      return result;
    }
    if ( v5 == 228 && v7 == 230 )
    {
      v14 = a2;
      v12 = sub_27797C0(a1);
    }
    else
    {
      result = v7 == 230 && v5 == 230;
      if ( !result )
        return result;
      v15 = sub_27797C0(a2);
      v14 = v16;
      v12 = v15;
    }
    v13 = (unsigned __int8 *)sub_27797C0(v14);
    return sub_2778F20(v13, v12);
  }
  v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v11 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v12 = *(_QWORD *)(a1 + 32 * (2 - v10));
  v13 = *(unsigned __int8 **)(a2 + 32 * (2 - v11));
  if ( (unsigned __int8 *)v12 != v13 || *(_QWORD *)(a2 + 32 * (3 - v11)) != *(_QWORD *)(a1 + 32 * (3 - v10)) )
  {
    if ( (unsigned int)**(unsigned __int8 **)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) - 12 > 1 )
      return 0;
    return sub_2778F20(v13, v12);
  }
  return result;
}
