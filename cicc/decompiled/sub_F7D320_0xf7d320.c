// Function: sub_F7D320
// Address: 0xf7d320
//
unsigned __int64 __fastcall sub_F7D320(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx
  char v4; // al

  result = *a2;
  if ( (unsigned __int8)result > 0x36u )
  {
LABEL_4:
    if ( (unsigned int)(unsigned __int8)result - 48 > 1 && (unsigned __int8)(result - 55) > 1u )
    {
LABEL_7:
      if ( (_BYTE)result == 58 )
      {
        v4 = *(_BYTE *)a1 >> 3;
LABEL_9:
        result = a2[1] & 1 | (2 * ((a2[1] >> 1) & 0xFE | v4 & 1u));
        a2[1] = result;
        return result;
      }
      if ( (((_BYTE)result - 68) & 0xFB) != 0 )
        goto LABEL_12;
      goto LABEL_20;
    }
LABEL_6:
    sub_B448B0((__int64)a2, (*(_BYTE *)a1 & 4) != 0);
    result = *a2;
    goto LABEL_7;
  }
  v3 = 0x40540000000000LL;
  if ( _bittest64(&v3, result) )
  {
    sub_B447F0(a2, *(_BYTE *)a1 & 1);
    sub_B44850(a2, (*(_BYTE *)a1 & 2) != 0);
    result = *a2;
    goto LABEL_4;
  }
  if ( (unsigned int)(unsigned __int8)result - 48 <= 1 )
    goto LABEL_6;
  if ( (((_BYTE)result - 68) & 0xFB) != 0 )
    goto LABEL_14;
LABEL_20:
  sub_B448D0((__int64)a2, (*(_BYTE *)a1 & 0x10) != 0);
  result = *a2;
LABEL_12:
  if ( (_BYTE)result == 67 )
  {
    sub_B447F0(a2, *(_BYTE *)a1 & 1);
    sub_B44850(a2, (*(_BYTE *)a1 & 2) != 0);
    result = *a2;
  }
LABEL_14:
  if ( (_BYTE)result == 63 )
  {
    sub_B4DDE0((__int64)a2, *(_DWORD *)(a1 + 4));
    result = *a2;
  }
  if ( (_BYTE)result == 82 )
  {
    v4 = *(_BYTE *)a1 >> 5;
    goto LABEL_9;
  }
  return result;
}
