// Function: sub_25BC620
// Address: 0x25bc620
//
_QWORD *__fastcall sub_25BC620(_QWORD *a1, __int64 a2, __int64 **a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  _QWORD *v6; // r14
  _QWORD *v7; // rsi
  _QWORD *v8; // rdx

  v4 = sub_B6AC80((__int64)a3, 356);
  v5 = sub_B6AC80((__int64)a3, 357);
  v6 = (_QWORD *)v5;
  if ( v4 )
  {
    LOBYTE(v4) = sub_25BBDB0(a3, (_QWORD *)v4, 0);
    if ( !v6 )
      goto LABEL_4;
  }
  else if ( !v5 )
  {
    v7 = a1 + 4;
    v8 = a1 + 10;
    goto LABEL_8;
  }
  LOBYTE(v4) = sub_25BBDB0(a3, v6, 1) | v4;
LABEL_4:
  v7 = a1 + 4;
  v8 = a1 + 10;
  if ( (_BYTE)v4 )
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
LABEL_8:
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
