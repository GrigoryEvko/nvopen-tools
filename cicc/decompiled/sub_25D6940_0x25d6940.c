// Function: sub_25D6940
// Address: 0x25d6940
//
_QWORD *__fastcall sub_25D6940(_QWORD *a1, __m128i a2, __int64 a3, __int64 a4)
{
  char v4; // al
  _QWORD *v5; // rsi
  _QWORD *v6; // rdx
  _BYTE v8[9]; // [rsp+Fh] [rbp-11h] BYREF

  v4 = sub_25D61C0(a4, (__int64)sub_25CCE60, (__int64)v8, a2);
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
