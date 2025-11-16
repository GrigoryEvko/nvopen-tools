// Function: sub_2BDBFE0
// Address: 0x2bdbfe0
//
bool __fastcall sub_2BDBFE0(_QWORD *a1, __int64 a2, unsigned __int16 a3, char a4)
{
  __int64 v6; // r12
  bool result; // al
  char v8; // al
  __int64 (__fastcall *v9)(__int64, unsigned int); // rdx

  v6 = sub_222F790(a1, a2);
  result = 1;
  if ( (*(_WORD *)(*(_QWORD *)(v6 + 48) + 2LL * (unsigned __int8)a2) & a3) == 0 )
  {
    result = 0;
    if ( (a4 & 1) != 0 )
    {
      if ( *(_BYTE *)(v6 + 56) )
      {
        v8 = *(_BYTE *)(v6 + 152);
      }
      else
      {
        sub_2216D60(v6);
        v9 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v6 + 48LL);
        v8 = 95;
        if ( v9 != sub_CE72A0 )
          v8 = v9(v6, 95u);
      }
      return (_BYTE)a2 == (unsigned __int8)v8;
    }
  }
  return result;
}
