// Function: sub_731E70
// Address: 0x731e70
//
__int64 __fastcall sub_731E70(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = sub_731E00(a1);
  if ( (_DWORD)result )
    goto LABEL_5;
  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result == 3 )
  {
    v3 = *(_QWORD *)(a1 + 56);
    if ( (*(_BYTE *)(v3 + 170) & 0x40) == 0 )
      return sub_72FA80(v3, a2);
    goto LABEL_5;
  }
  if ( (_BYTE)result == 22 )
  {
    result = sub_8DC060(*(_QWORD *)(a1 + 56));
    if ( (_DWORD)result )
    {
LABEL_5:
      *(_DWORD *)(a2 + 80) = 1;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
  return result;
}
