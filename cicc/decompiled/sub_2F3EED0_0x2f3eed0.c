// Function: sub_2F3EED0
// Address: 0x2f3eed0
//
_QWORD *__fastcall sub_2F3EED0(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  char v7; // al
  _QWORD *v8; // rsi
  _QWORD *v9; // rdx
  __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h] BYREF
  __int64 v13[5]; // [rsp+10h] [rbp-50h] BYREF
  char v14; // [rsp+38h] [rbp-28h]

  v5 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v14 = 1;
  v11 = v5;
  v12 = v5;
  v13[0] = *a2;
  v13[1] = (__int64)sub_2F3BE80;
  v13[2] = (__int64)&v11;
  v13[3] = (__int64)sub_2F3BEA0;
  v13[4] = (__int64)&v12;
  v7 = sub_2F3E6C0(v13, a3, v6);
  v8 = a1 + 4;
  v9 = a1 + 10;
  if ( v7 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v8;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v9;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
  else
  {
    a1[1] = v8;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[4] = &qword_4F82400;
    a1[7] = v9;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    *a1 = 1;
    return a1;
  }
}
