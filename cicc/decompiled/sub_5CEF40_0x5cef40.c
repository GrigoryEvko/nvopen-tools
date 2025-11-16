// Function: sub_5CEF40
// Address: 0x5cef40
//
__int64 __fastcall sub_5CEF40(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  _QWORD **v4; // rdi

  if ( *(_BYTE *)(a1 + 140) == 12 && *(_BYTE *)(a1 + 184) == 8 )
  {
    v4 = (_QWORD **)(a1 + 104);
    if ( *(_QWORD *)(a1 + 104) )
      v4 = sub_5CB9F0(v4);
    *v4 = a2;
    return a1;
  }
  else
  {
    result = sub_7259C0(12);
    *(_QWORD *)(result + 160) = a1;
    *(_BYTE *)(result + 184) = 8;
    *(_QWORD *)(result + 104) = a2;
  }
  return result;
}
