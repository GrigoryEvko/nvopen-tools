// Function: sub_B4BB80
// Address: 0xb4bb80
//
_QWORD *__fastcall sub_B4BB80(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, unsigned __int16 a6)
{
  __int64 v10; // rax
  _QWORD *result; // rax
  __int64 v12; // rbx
  __int64 v13; // rax

  v10 = sub_BCB120(a2);
  result = sub_B44260(a1, v10, 1, a4, a5, a6);
  if ( a3 )
  {
    v12 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v12 )
    {
      v13 = *(_QWORD *)(v12 + 8);
      **(_QWORD **)(v12 + 16) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
    }
    result = *(_QWORD **)(a3 + 16);
    *(_QWORD *)v12 = a3;
    *(_QWORD *)(v12 + 8) = result;
    if ( result )
      result[2] = v12 + 8;
    *(_QWORD *)(v12 + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v12;
  }
  return result;
}
