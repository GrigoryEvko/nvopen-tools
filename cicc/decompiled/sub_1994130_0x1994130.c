// Function: sub_1994130
// Address: 0x1994130
//
__int64 __fastcall sub_1994130(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v6; // al
  __int64 v8; // rax
  unsigned int v9; // eax

  v6 = *(_BYTE *)(a2 + 16);
  LOBYTE(a5) = v6 == 54;
  if ( v6 == 55 )
  {
    LOBYTE(a5) = *(_QWORD *)(a2 - 24) == a3;
    return a5;
  }
  if ( v6 != 78 )
  {
    if ( v6 == 59 )
    {
      LOBYTE(a5) = *(_QWORD *)(a2 - 48) == a3;
    }
    else if ( v6 == 58 )
    {
      LOBYTE(a5) = *(_QWORD *)(a2 - 72) == a3;
    }
    return a5;
  }
  v8 = *(_QWORD *)(a2 - 24);
  a5 = 0;
  if ( *(_BYTE *)(v8 + 16) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
    return a5;
  v9 = *(_DWORD *)(v8 + 36);
  if ( v9 != 137 )
  {
    if ( v9 <= 0x89 )
    {
      if ( (v9 & 0xFFFFFFFD) == 0x85 )
      {
        a5 = 1;
        if ( a3 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
          LOBYTE(a5) = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))) == a3;
        return a5;
      }
      goto LABEL_12;
    }
    if ( v9 != 148 )
    {
LABEL_12:
      a5 = sub_14A36E0(a1);
      if ( (_BYTE)a5 )
        LOBYTE(a5) = a3 == 0;
      return a5;
    }
  }
  LOBYTE(a5) = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) == a3;
  return a5;
}
