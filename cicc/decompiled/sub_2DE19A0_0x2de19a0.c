// Function: sub_2DE19A0
// Address: 0x2de19a0
//
_QWORD *__fastcall sub_2DE19A0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  char v4; // bl
  _QWORD *v5; // rsi
  _QWORD *v6; // rdx
  int v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v9; // [rsp+8h] [rbp-28h]
  __int64 v10; // [rsp+10h] [rbp-20h]

  v3 = *a2;
  v8 = 0;
  v9 = 0;
  v10 = v3;
  v4 = sub_2DE1890((__int64)&v8, a3);
  if ( v9 )
    sub_2DDBD80(v9);
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
  }
  return a1;
}
