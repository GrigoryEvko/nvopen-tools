// Function: sub_2FACA00
// Address: 0x2faca00
//
_QWORD *__fastcall sub_2FACA00(_QWORD *a1, __int64 **a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 **v5; // rsi
  char v6; // al
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx
  __int64 *v10[22]; // [rsp+0h] [rbp-B0h] BYREF

  v4 = *a2;
  v5 = *(__int64 ***)(a3 + 40);
  memset(v10, 0, 128);
  v10[16] = v4;
  sub_2FA8740(v10, v5);
  v6 = sub_2FAC800((__int64)v10, a3);
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
  }
  else
  {
    a1[1] = v7;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v8;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}
