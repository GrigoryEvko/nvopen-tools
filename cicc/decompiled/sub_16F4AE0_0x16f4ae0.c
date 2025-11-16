// Function: sub_16F4AE0
// Address: 0x16f4ae0
//
_BYTE *__fastcall sub_16F4AE0(__int64 a1, __int64 a2, char a3, char a4)
{
  __int64 v4; // rax
  _BYTE *result; // rax

  v4 = *(_QWORD *)(a2 + 8);
  if ( v4 == *(_QWORD *)(a2 + 16) )
  {
    *(_QWORD *)a1 = 0;
    v4 = 0;
  }
  else
  {
    *(_QWORD *)a1 = a2;
  }
  *(_BYTE *)(a1 + 8) = a4;
  *(_BYTE *)(a1 + 9) = a3;
  *(_DWORD *)(a1 + 12) = 1;
  *(_QWORD *)(a1 + 16) = v4;
  *(_QWORD *)(a1 + 24) = 0;
  result = *(_BYTE **)(a2 + 8);
  if ( *(_BYTE **)(a2 + 16) != result && (a3 || *result != 10 && (*result != 13 || result[1] != 10)) )
    return (_BYTE *)sub_16F48F0(a1);
  return result;
}
