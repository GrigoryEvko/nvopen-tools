// Function: sub_945E50
// Address: 0x945e50
//
__int64 __fastcall sub_945E50(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // rsi
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rax

  v5 = *(_QWORD *)(a1 + 96);
  if ( v5 )
  {
    v6 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 + 48 == v6 )
    {
      v7 = *(_QWORD *)(a1 + 200);
    }
    else
    {
      if ( !v6 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 <= 0xA )
        sub_91B8A0("unexpected: last basic block has terminator!", a2, 1);
      v7 = *(_QWORD *)(a1 + 200);
      if ( *(_QWORD *)(v7 + 16) )
      {
        v8 = *(_QWORD **)(a1 + 200);
        return sub_92FEA0(a1, v8, 0);
      }
    }
    result = sub_BD84D0(v7, v5);
    v11 = *(_QWORD *)(a1 + 200);
    if ( v11 )
      goto LABEL_13;
  }
  else
  {
    v8 = *(_QWORD **)(a1 + 200);
    v10 = v8[2];
    if ( !v10 )
      return sub_92FEA0(a1, v8, 0);
    if ( *(_QWORD *)(v10 + 8) )
      return sub_92FEA0(a1, v8, 0);
    v12 = *(_QWORD *)(v10 + 24);
    if ( *(_BYTE *)v12 != 31 )
      return sub_92FEA0(a1, v8, 0);
    if ( (*(_DWORD *)(v12 + 4) & 0x7FFFFFF) != 1 )
      return sub_92FEA0(a1, v8, 0);
    v13 = *(_QWORD **)(v12 - 32);
    if ( v8 != v13 || !v13 )
      return sub_92FEA0(a1, v8, 0);
    v14 = *(_QWORD *)(v12 + 40);
    *(_WORD *)(a1 + 112) = 0;
    *(_QWORD *)(a1 + 96) = v14;
    *(_QWORD *)(a1 + 104) = v14 + 48;
    result = sub_B43D60(v12, v8, a3, a4);
    v11 = *(_QWORD *)(a1 + 200);
    if ( v11 )
    {
LABEL_13:
      sub_AA5290(v11);
      return j_j___libc_free_0(v11, 80);
    }
  }
  return result;
}
