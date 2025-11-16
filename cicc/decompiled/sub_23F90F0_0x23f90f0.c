// Function: sub_23F90F0
// Address: 0x23f90f0
//
_QWORD *__fastcall sub_23F90F0(_QWORD *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  char v8; // al
  _QWORD *v9; // rsi
  _QWORD *v10; // rdx

  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v7 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v8 = sub_23F7470(a3, v6 + 8, v7 + 8, a2);
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
  }
  else
  {
    a1[1] = v9;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v10;
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
