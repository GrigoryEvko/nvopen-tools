// Function: sub_2DC1A10
// Address: 0x2dc1a10
//
_QWORD *__fastcall sub_2DC1A10(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  char v10; // al
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx

  v4 = *a2;
  v5 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
  if ( v5 == sub_23CE270 )
    BUG();
  v7 = 0;
  v8 = ((__int64 (__fastcall *)(__int64, __int64))v5)(v4, a3);
  v9 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
  if ( v9 != sub_2C8F680 )
    v7 = ((__int64 (__fastcall *)(__int64, _QWORD))v9)(v8, 0);
  v10 = sub_2DBF120(a3, v7);
  v11 = a1 + 4;
  v12 = a1 + 10;
  if ( v10 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v11;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v12;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v11;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v12;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
