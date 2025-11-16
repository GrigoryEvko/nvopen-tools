// Function: sub_19695C0
// Address: 0x19695c0
//
__int64 __fastcall sub_19695C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD *v4; // rax
  bool v5; // zf

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 77 && *(_QWORD *)(a1 + 40) == a3 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v4 = *(_QWORD **)(a1 - 8);
    else
      v4 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( a2 == *v4 )
    {
      return a1;
    }
    else
    {
      v5 = a2 == v4[3];
      result = 0;
      if ( v5 )
        return a1;
    }
  }
  return result;
}
