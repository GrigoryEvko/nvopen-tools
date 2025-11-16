// Function: sub_2CF5710
// Address: 0x2cf5710
//
_QWORD *__fastcall sub_2CF5710(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rcx
  char v10; // al
  char v11; // al
  _QWORD *v12; // rsi
  _QWORD *v13; // rdx
  __int16 v15; // [rsp+Eh] [rbp-32h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v7 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v8 = *a2 == 0;
  v9 = v7 + 8;
  LOBYTE(v15) = a2[1];
  v10 = byte_50145E8;
  if ( v8 )
    v10 = 1;
  HIBYTE(v15) = v10;
  v11 = sub_2CF5350(&v15, a3, v6, v9);
  v12 = a1 + 4;
  v13 = a1 + 10;
  if ( v11 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v12;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v13;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v12;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v13;
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
