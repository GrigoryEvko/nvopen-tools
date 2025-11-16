// Function: sub_73DDB0
// Address: 0x73ddb0
//
_BYTE *__fastcall sub_73DDB0(_QWORD *a1)
{
  _BYTE *result; // rax
  __int64 v2; // rsi

  if ( !*((_BYTE *)a1 + 24) )
    return a1;
  if ( (unsigned int)sub_8D32E0(*a1) )
  {
    v2 = sub_8D46C0(*a1);
  }
  else if ( dword_4F077C4 == 2 && (unsigned int)sub_8D3D40(*a1) )
  {
    v2 = dword_4D03B80;
  }
  else
  {
    v2 = sub_72C930();
  }
  a1[2] = 0;
  result = sub_73DC30(4u, v2, (__int64)a1);
  result[27] |= 2u;
  return result;
}
