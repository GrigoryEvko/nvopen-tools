// Function: sub_BDBD80
// Address: 0xbdbd80
//
_BYTE *__fastcall sub_BDBD80(__int64 a1, _BYTE *a2)
{
  bool v4; // cc
  _BYTE *v5; // rsi
  _BYTE *v6; // rdi
  _BYTE *result; // rax

  v4 = *a2 <= 0x1Cu;
  v5 = *(_BYTE **)a1;
  if ( v4 )
  {
    sub_A5C020(a2, (__int64)v5, 1, a1 + 16);
    v6 = *(_BYTE **)a1;
    result = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned __int64)result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      goto LABEL_3;
  }
  else
  {
    sub_A693B0((__int64)a2, v5, a1 + 16, 0);
    v6 = *(_BYTE **)a1;
    result = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned __int64)result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
    {
LABEL_3:
      *((_QWORD *)v6 + 4) = result + 1;
      *result = 10;
      return result;
    }
  }
  return (_BYTE *)sub_CB5D20(v6, 10);
}
