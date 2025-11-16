// Function: sub_15F7090
// Address: 0x15f7090
//
__int64 __fastcall sub_15F7090(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 result; // rax
  __int64 v13; // rax

  if ( a3 )
  {
    v6 = sub_1643270(a2);
    sub_15F1F50(a1, v6, 1, a1 - 24, 1, a4);
    v7 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *v7 )
    {
      v8 = v7[1];
      v9 = v7[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v9 = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(v8 + 16) & 3LL | v9;
    }
    v10 = *(_QWORD *)(a3 + 8);
    *v7 = a3;
    v7[1] = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(v10 + 16) & 3LL;
    v11 = v7[2];
    *(_QWORD *)(a3 + 8) = v7;
    result = v11 & 3;
    v7[2] = result | (a3 + 8);
  }
  else
  {
    v13 = sub_1643270(a2);
    return sub_15F1F50(a1, v13, 1, a1, 0, a4);
  }
  return result;
}
