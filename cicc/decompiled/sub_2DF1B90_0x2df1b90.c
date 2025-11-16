// Function: sub_2DF1B90
// Address: 0x2df1b90
//
_QWORD *__fastcall sub_2DF1B90(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 (*v11)(); // rax
  __int64 v12; // rdi
  __int64 (*v13)(); // rdx
  __int64 v14; // rax
  char v15; // al
  _QWORD *v16; // rsi
  _QWORD *v17; // rdx
  __int64 v19; // [rsp+8h] [rbp-68h]
  _QWORD v20[12]; // [rsp+10h] [rbp-60h] BYREF

  v19 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v7 = *(_QWORD *)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v9 = *a2;
  v20[0] = a3;
  v20[2] = v7;
  v10 = v8 + 8;
  v20[1] = v19 + 8;
  v11 = *(__int64 (**)())(*(_QWORD *)v9 + 16LL);
  if ( v11 == sub_23CE270 )
    BUG();
  v12 = ((__int64 (__fastcall *)(__int64, __int64))v11)(v9, a3);
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 144LL);
  v14 = 0;
  if ( v13 != sub_2C8F680 )
    v14 = ((__int64 (__fastcall *)(__int64))v13)(v12);
  v20[3] = v14;
  v20[4] = v10;
  v15 = sub_2DF0A30((__int64)v20);
  v16 = a1 + 4;
  v17 = a1 + 10;
  if ( v15 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v16;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v17;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v16;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v17;
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
