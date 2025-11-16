// Function: sub_10C54C0
// Address: 0x10c54c0
//
__int64 __fastcall sub_10C54C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v5; // r12
  _BYTE *v6; // r12
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rsi

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 == 58 )
  {
    v5 = *(_BYTE **)(a2 - 64);
    if ( *v5 == 59
      && ((v10 = sub_995B10((_QWORD **)a1, *((_QWORD *)v5 - 8)), v11 = *((_QWORD *)v5 - 4), v10)
       && v11 == *(_QWORD *)(a1 + 8)
       || (unsigned __int8)sub_995B10((_QWORD **)a1, v11) && *((_QWORD *)v5 - 8) == *(_QWORD *)(a1 + 8)) )
    {
      v6 = *(_BYTE **)(a2 - 32);
      if ( v6 )
      {
        **(_QWORD **)(a1 + 16) = v6;
        return 1;
      }
    }
    else
    {
      v6 = *(_BYTE **)(a2 - 32);
    }
    if ( *v6 == 59 )
    {
      if ( (v7 = sub_995B10((_QWORD **)a1, *((_QWORD *)v6 - 8)), v8 = *((_QWORD *)v6 - 4), v7)
        && v8 == *(_QWORD *)(a1 + 8)
        || (unsigned __int8)sub_995B10((_QWORD **)a1, v8) && *((_QWORD *)v6 - 8) == *(_QWORD *)(a1 + 8) )
      {
        v9 = *(_QWORD *)(a2 - 64);
        if ( v9 )
        {
          **(_QWORD **)(a1 + 16) = v9;
          return 1;
        }
      }
    }
  }
  return 0;
}
