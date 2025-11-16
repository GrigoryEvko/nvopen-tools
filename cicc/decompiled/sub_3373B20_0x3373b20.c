// Function: sub_3373B20
// Address: 0x3373b20
//
_BYTE *__fastcall sub_3373B20(__int64 a1)
{
  __int64 v1; // rax
  int v2; // eax
  unsigned int v3; // edx
  bool v4; // cc
  _BYTE *result; // rax

  v1 = sub_B2E500(**(_QWORD **)(a1 + 960));
  v2 = sub_B2A630(v1);
  v3 = v2 - 9;
  v4 = (unsigned int)(v2 - 7) <= 1;
  result = *(_BYTE **)(*(_QWORD *)(a1 + 960) + 744LL);
  if ( v4 )
  {
    result[234] = 1;
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 40LL) + 578LL) = 1;
  }
  else
  {
    result[233] = 1;
  }
  if ( v3 <= 1 )
    result[235] = 1;
  return result;
}
