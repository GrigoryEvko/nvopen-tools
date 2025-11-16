// Function: sub_25ADD60
// Address: 0x25add60
//
_QWORD *__fastcall sub_25ADD60(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // r14
  _QWORD *v5; // rbx
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a1 + 4;
  v5 = a1 + 10;
  v8[0] = *a3;
  v7 = sub_B8C320(v8);
  if ( sub_BA91D0((__int64)a3, "Cross-DSO CFI", 0xDu) )
  {
    sub_25AC8F0(&v7, (__int64)a3);
    memset(a1, 0, 0x60u);
    a1[1] = v4;
    a1[7] = v5;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v4;
    a1[2] = 0x100000002LL;
    a1[7] = v5;
    a1[4] = &qword_4F82400;
    a1[6] = 0;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
