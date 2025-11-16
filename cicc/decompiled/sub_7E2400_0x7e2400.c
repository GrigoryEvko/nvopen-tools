// Function: sub_7E2400
// Address: 0x7e2400
//
_BYTE *__fastcall sub_7E2400(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // rsi
  _BYTE *result; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // r12
  __int64 v17; // rax
  _QWORD *v18; // rax

  v9 = sub_7E05E0(a2, a3);
  if ( !v9 || (v12 = v9, (*(_BYTE *)(v9 + 144) & 4) != 0) )
  {
    if ( a3 )
    {
      v15 = sub_73E1B0((__int64)a1, a3);
      v16 = sub_7E23D0(v15);
      *((_QWORD *)v16 + 2) = sub_73A830(a3, byte_4F06A51[0]);
      v17 = sub_7E1C30();
      v18 = sub_73DBF0(0x32u, v17, (__int64)v16);
      a1 = sub_73DCD0(v18);
    }
    return sub_73DC90(a1, a4);
  }
  else
  {
    if ( a3 || (v13 = *(_QWORD *)(v9 + 120), v10 == v13) || (unsigned int)sub_8D97D0(v10, v13, 1, v10, v11) )
      result = sub_73DE50((__int64)a1, v12);
    else
      result = a1;
    if ( a4 != *(_QWORD *)(v12 + 120) )
      return sub_73DC90(result, a4);
  }
  return result;
}
