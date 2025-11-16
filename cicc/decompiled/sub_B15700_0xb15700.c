// Function: sub_B15700
// Address: 0xb15700
//
_DWORD *__fastcall sub_B15700(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _DWORD *result; // rax
  bool v5; // zf
  unsigned __int8 v6; // dl
  void **v7; // rax

  *(_BYTE *)(a1 + 12) = a4;
  *(_DWORD *)(a1 + 8) = 2;
  result = &unk_49D9B88;
  v5 = *(_QWORD *)(a2 + 48) == 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)a1 = &unk_49D9B88;
  *(_QWORD *)(a1 + 24) = a3;
  *(_QWORD *)(a1 + 32) = a2;
  if ( !v5 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    result = (_DWORD *)sub_B91F50(a2, "srcloc", 6);
    if ( result )
    {
      v6 = *((_BYTE *)result - 16);
      if ( (v6 & 2) != 0 )
      {
        if ( !*(result - 6) )
          return result;
        v7 = (void **)*((_QWORD *)result - 4);
      }
      else
      {
        if ( (*(_WORD *)(result - 4) & 0x3C0) == 0 )
          return result;
        v7 = (void **)&result[-2 * ((v6 >> 2) & 0xF) - 4];
      }
      result = *v7;
      if ( *(_BYTE *)result == 1 )
      {
        result = (_DWORD *)*((_QWORD *)result + 17);
        if ( *(_BYTE *)result == 17 )
        {
          if ( result[8] <= 0x40u )
            result = (_DWORD *)*((_QWORD *)result + 3);
          else
            result = (_DWORD *)**((_QWORD **)result + 3);
          *(_QWORD *)(a1 + 16) = result;
        }
      }
    }
  }
  return result;
}
