// Function: sub_73DCD0
// Address: 0x73dcd0
//
_BYTE *__fastcall sub_73DCD0(_QWORD *a1)
{
  _QWORD *v1; // r12
  char v2; // al
  _BYTE *result; // rax
  __int64 v4; // rsi

  v1 = a1;
  v2 = *((_BYTE *)a1 + 24);
  if ( !v2 )
    return v1;
  if ( v2 == 1 && ((*((_BYTE *)a1 + 27) & 2) != 0 || dword_4D03F94) && !*((_BYTE *)a1 + 56) )
    return (_BYTE *)a1[9];
  if ( (unsigned int)sub_8D2E30(*a1) || (unsigned int)sub_8D2FB0(*a1) && dword_4D03F94 )
  {
    v4 = sub_8D46C0(*a1);
  }
  else if ( dword_4F077C4 == 2 && (unsigned int)sub_8D3D40(*a1) )
  {
    v4 = dword_4D03B80;
  }
  else
  {
    v4 = sub_72C930();
  }
  a1[2] = 0;
  result = sub_73DC30(3u, v4, (__int64)a1);
  result[27] |= 2u;
  return result;
}
