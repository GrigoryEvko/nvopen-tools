// Function: sub_27848A0
// Address: 0x27848a0
//
_QWORD *__fastcall sub_27848A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // r15
  __int64 v6; // r14
  char v7; // al
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx

  v4 = 0;
  v6 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  while ( 1 )
  {
    v7 = sub_2784530(a3, v6);
    if ( !v7 )
      break;
    v4 = v7;
    sub_F62E00(a3, 0, 0, v8, v9, v10);
  }
  v11 = a1 + 4;
  v12 = a1 + 10;
  if ( v4 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v11;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v12;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v11;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v12;
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
