// Function: sub_27A2DA0
// Address: 0x27a2da0
//
__int64 __fastcall sub_27A2DA0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE **v3; // r13
  __int64 v4; // rax
  _BYTE **v5; // rbx
  _BYTE *v6; // r12
  unsigned int v7; // r15d

  v3 = (_BYTE **)a2;
  v4 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v5 = *(_BYTE ***)(a2 - 8);
    v3 = &v5[v4];
  }
  else
  {
    v5 = (_BYTE **)(a2 - v4 * 8);
  }
  if ( v5 == v3 )
  {
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v6 = *v5;
      if ( **v5 > 0x1Cu )
      {
        v7 = sub_B19720(*(_QWORD *)(a1 + 216), *((_QWORD *)v6 + 5), a3);
        if ( !(_BYTE)v7 && (*v6 != 63 || !(unsigned __int8)sub_27A2DA0(a1, v6, a3)) )
          break;
      }
      v5 += 4;
      if ( v3 == v5 )
        return 1;
    }
  }
  return v7;
}
