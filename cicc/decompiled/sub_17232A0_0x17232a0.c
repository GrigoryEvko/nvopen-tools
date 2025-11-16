// Function: sub_17232A0
// Address: 0x17232a0
//
__int64 __fastcall sub_17232A0(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 35 )
  {
    result = sub_1722F40(a1, *(_QWORD *)(a2 - 48));
    v5 = *(_QWORD *)(a2 - 24);
    if ( !(_BYTE)result || !v5 )
    {
      result = sub_1722F40(a1, v5);
      if ( !(_BYTE)result )
        return 0;
      v6 = *(_QWORD *)(a2 - 48);
      if ( !v6 )
        return 0;
LABEL_8:
      *a1[4] = v6;
      return result;
    }
  }
  else
  {
    if ( v2 != 5 || *(_WORD *)(a2 + 18) != 11 )
      return 0;
    result = sub_17230F0(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !(_BYTE)result )
    {
      v5 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      goto LABEL_13;
    }
    v5 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v5 )
    {
LABEL_13:
      result = sub_17230F0(a1, v5);
      if ( !(_BYTE)result )
        return 0;
      v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( !v6 )
        return 0;
      goto LABEL_8;
    }
  }
  *a1[4] = v5;
  return result;
}
