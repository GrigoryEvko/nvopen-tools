// Function: sub_2A2C590
// Address: 0x2a2c590
//
_QWORD *__fastcall sub_2A2C590(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  char v9; // al
  _QWORD *v10; // rsi
  _QWORD *v11; // rdx
  __int64 v13; // [rsp+8h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v13 = sub_BC1CD0(a4, &unk_4F86D28, a3);
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v9 = sub_2A2B740(v7 + 8, v13 + 8, v8 + 8, v6 + 8);
  v10 = a1 + 4;
  v11 = a1 + 10;
  if ( v9 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v10;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v11;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v10;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v11;
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
