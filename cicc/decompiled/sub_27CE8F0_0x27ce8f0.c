// Function: sub_27CE8F0
// Address: 0x27ce8f0
//
__int64 __fastcall sub_27CE8F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax

  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 && !(unsigned __int8)sub_DFA750(a1) )
    return 0;
  v8 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( a4 != *(_QWORD *)v8 )
    return 0;
  if ( a4 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    **(_QWORD **)(v8 + 16) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v8 + 16);
  }
  *(_QWORD *)v8 = a5;
  result = 1;
  if ( a5 )
  {
    v10 = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(v8 + 8) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = v8 + 8;
    *(_QWORD *)(v8 + 16) = a5 + 16;
    *(_QWORD *)(a5 + 16) = v8;
    return 1;
  }
  return result;
}
