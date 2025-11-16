// Function: sub_2A86430
// Address: 0x2a86430
//
_QWORD *__fastcall sub_2A86430(_QWORD *a1, unsigned __int8 *a2, _QWORD *a3)
{
  char v4; // bl
  char v5; // al
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx

  v4 = sub_2A861E0((__int64)a3);
  v5 = sub_2A85BE0(a3, *a2);
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
