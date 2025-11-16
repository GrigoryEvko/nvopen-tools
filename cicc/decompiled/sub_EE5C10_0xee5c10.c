// Function: sub_EE5C10
// Address: 0xee5c10
//
_QWORD *__fastcall sub_EE5C10(_QWORD *a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rbx
  void *v3; // rdx
  _QWORD *v4; // rax

  result = (_QWORD *)sub_22077B0(1472);
  v2 = result;
  if ( result )
  {
    v3 = result + 5;
    *result = 0;
    v4 = result + 37;
    *(v4 - 35) = v3;
    *(v4 - 34) = v3;
    *(v4 - 36) = 0;
    v2[4] = v4;
    memset(v3, 0, 0x100u);
    v2[37] = v2 + 40;
    v2[39] = v2 + 72;
    v2[38] = v2 + 40;
    memset(v2 + 40, 0, 0x100u);
    *(_OWORD *)(v2 + 75) = 0;
    v2[72] = v2 + 75;
    v2[73] = v2 + 75;
    v2[74] = v2 + 83;
    v2[83] = v2 + 86;
    v2[84] = v2 + 86;
    v2[85] = v2 + 90;
    v2[90] = v2 + 93;
    v2[91] = v2 + 93;
    v2[92] = v2 + 97;
    *((_WORD *)v2 + 388) = 1;
    *(_OWORD *)(v2 + 77) = 0;
    *(_OWORD *)(v2 + 79) = 0;
    *(_OWORD *)(v2 + 81) = 0;
    *((_OWORD *)v2 + 43) = 0;
    *((_OWORD *)v2 + 44) = 0;
    *(_OWORD *)(v2 + 93) = 0;
    *(_OWORD *)(v2 + 95) = 0;
    *((_BYTE *)v2 + 778) = 0;
    v2[98] = -1;
    v2[99] = 0;
    *((_DWORD *)v2 + 200) = 0;
    v2[101] = 0;
    v2[103] = v2 + 105;
    v2[104] = 0x400000000LL;
    v2[109] = v2 + 111;
    v2[102] = 0;
    v2[110] = 0;
    v2[111] = 0;
    v2[112] = 1;
    sub_C656D0((__int64)(v2 + 113), 6);
    v2[115] = 0;
    result = v2 + 120;
    v2[116] = 0;
    v2[118] = 0;
    v2[119] = 1;
    *((_WORD *)v2 + 468) = 256;
    do
    {
      if ( result )
        *result = -4096;
      result += 2;
    }
    while ( v2 + 184 != result );
  }
  *a1 = v2;
  return result;
}
