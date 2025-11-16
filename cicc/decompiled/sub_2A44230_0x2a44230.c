// Function: sub_2A44230
// Address: 0x2a44230
//
_QWORD *__fastcall sub_2A44230(_QWORD *a1, __int64 a2, __int64 a3)
{
  char v3; // al
  _QWORD *v4; // rsi
  _QWORD *v5; // rdx

  v3 = sub_2A43B20(a3);
  v4 = a1 + 4;
  v5 = a1 + 10;
  if ( v3 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v4;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v5;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v4;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v5;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
