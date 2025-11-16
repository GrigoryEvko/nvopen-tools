// Function: sub_2D4F890
// Address: 0x2d4f890
//
_QWORD *__fastcall sub_2D4F890(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdx
  char v6; // al
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx
  _QWORD v10[3]; // [rsp+0h] [rbp-20h] BYREF

  v5 = *a2;
  v10[0] = 0;
  v10[1] = 0;
  v6 = sub_2D4F770((__int64)v10, a3, v5);
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
