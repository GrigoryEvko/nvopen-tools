// Function: sub_36D4160
// Address: 0x36d4160
//
_QWORD *__fastcall sub_36D4160(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  char v4; // bl
  char v5; // al
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx

  v4 = sub_36D4010(a3, (__int64)"llvm.global_ctors", 0x11u, 1);
  v5 = sub_36D4010(a3, (__int64)"llvm.global_dtors", 0x11u, 0);
  v6 = a1 + 4;
  v7 = a1 + 10;
  if ( v4 || v5 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v6;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v7;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v6;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v7;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
