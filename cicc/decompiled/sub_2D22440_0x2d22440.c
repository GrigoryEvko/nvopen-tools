// Function: sub_2D22440
// Address: 0x2d22440
//
_QWORD *__fastcall sub_2D22440(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v5; // al
  _QWORD *v6; // rsi
  _QWORD *v7; // rdx
  unsigned int *v9; // [rsp+8h] [rbp-18h] BYREF

  v9 = (unsigned int *)(sub_BC0510(a4, &unk_5035D48, a3) + 8);
  v5 = sub_2D221B0(&v9, a3);
  v6 = a1 + 4;
  v7 = a1 + 10;
  if ( v5 )
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
