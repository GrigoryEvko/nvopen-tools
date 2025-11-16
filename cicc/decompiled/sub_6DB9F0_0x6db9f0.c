// Function: sub_6DB9F0
// Address: 0x6db9f0
//
__int64 __fastcall sub_6DB9F0(__int64 *a1, _QWORD *a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 v7; // r14
  unsigned __int8 v8; // al
  __int64 result; // rax

  v7 = *((_WORD *)a1 + 4);
  v8 = *(_BYTE *)(*a1 + 56);
  if ( v8 == 108 )
  {
    *((_WORD *)a1 + 4) = 147;
  }
  else
  {
    if ( v8 <= 0x6Cu )
    {
      if ( v8 == 106 )
      {
        *((_WORD *)a1 + 4) = 29;
        goto LABEL_6;
      }
      if ( v8 == 107 )
      {
        *((_WORD *)a1 + 4) = 30;
LABEL_6:
        sub_6E59E0(a2);
        result = sub_6D7FC0(0, (__int64)a1, 1, 0, a2, a3);
        *((_WORD *)a1 + 4) = v7;
        return result;
      }
LABEL_12:
      sub_721090(a1);
    }
    if ( v8 != 109 )
      goto LABEL_12;
    *((_WORD *)a1 + 4) = 148;
  }
  result = sub_6B0A80(0, a1, 1, (__int64)a2, a3, a6);
  *((_WORD *)a1 + 4) = v7;
  return result;
}
