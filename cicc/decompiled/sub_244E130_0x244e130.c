// Function: sub_244E130
// Address: 0x244e130
//
_QWORD *__fastcall sub_244E130(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r9
  char v8; // al
  _QWORD *v9; // rsi
  _QWORD *v10; // rdx

  v6 = sub_BC0510(a4, &unk_4F87C68, a3);
  v8 = sub_244AB10(a3, v6 + 8, (unsigned __int8)qword_4FE5FE8 | *a2, byte_4FE5F08 | a2[1], a4, v7);
  v9 = a1 + 4;
  v10 = a1 + 10;
  if ( v8 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v9;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v9;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v10;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
