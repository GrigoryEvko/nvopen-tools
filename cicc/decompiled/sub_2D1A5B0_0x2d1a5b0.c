// Function: sub_2D1A5B0
// Address: 0x2d1a5b0
//
_QWORD *__fastcall sub_2D1A5B0(_QWORD *a1, char *a2, __int64 a3)
{
  char v3; // al
  char v4; // al
  _QWORD *v5; // rsi
  _QWORD *v6; // rdx
  char v8[10]; // [rsp+Eh] [rbp-12h] BYREF

  v3 = a2[1];
  v8[0] = *a2;
  v8[1] = v3;
  v4 = sub_2D19B20(v8, a3);
  v5 = a1 + 4;
  v6 = a1 + 10;
  if ( v4 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v5;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v6;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v5;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v6;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
