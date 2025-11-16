// Function: sub_2CD2410
// Address: 0x2cd2410
//
_QWORD *__fastcall sub_2CD2410(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx
  _BYTE v10[9]; // [rsp+Fh] [rbp-11h] BYREF

  v10[0] = *a2;
  v6 = sub_2CCF450(v10, a3, a3, a4, (__int64)a2, a6);
  v7 = a1 + 4;
  v8 = a1 + 10;
  if ( v6 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v7;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v8;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v7;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v8;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
