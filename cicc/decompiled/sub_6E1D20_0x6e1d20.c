// Function: sub_6E1D20
// Address: 0x6e1d20
//
_DWORD *__fastcall sub_6E1D20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _DWORD *result; // rax
  char v6; // dl
  __int64 v7; // rsi
  __int64 v8; // rdi

  result = (_DWORD *)qword_4D03C50;
  v6 = *(_BYTE *)(qword_4D03C50 + 18LL);
  if ( (v6 & 0x40) == 0 )
  {
    v7 = *(unsigned __int8 *)(qword_4D03C50 + 17LL);
    if ( (v7 & 2) != 0 )
    {
      return (_DWORD *)sub_732910(a1, (((unsigned __int8)v7 >> 6) ^ 1) & 1, (v6 & 0x40) != 0);
    }
    else
    {
      result = &dword_4F04C44;
      if ( dword_4F04C44 == -1 )
      {
        result = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
        if ( (*((_BYTE *)result + 6) & 6) == 0 && *((_BYTE *)result + 4) != 12 )
        {
          v8 = *a1;
          if ( v8 )
            return (_DWORD *)sub_894C00(v8, v7, qword_4F04C68, a4, a5);
        }
      }
    }
  }
  return result;
}
