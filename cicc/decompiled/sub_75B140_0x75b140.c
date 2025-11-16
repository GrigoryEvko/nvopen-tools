// Function: sub_75B140
// Address: 0x75b140
//
_DWORD *__fastcall sub_75B140(
        __int64 (__fastcall *a1)(_QWORD, _QWORD),
        __int64 (__fastcall *a2)(_QWORD, _QWORD, _QWORD),
        __int64 (__fastcall *a3)(_QWORD, _QWORD),
        __int64 (__fastcall *a4)(_QWORD, _QWORD),
        __int64 (__fastcall *a5)(_QWORD, _QWORD),
        int a6,
        __int64 *a7,
        unsigned __int8 a8)
{
  __int64 (__fastcall *v9)(_QWORD, _QWORD); // rax
  __int64 *v10; // rdi
  __int64 (__fastcall *v11)(_QWORD, _QWORD); // r15
  __int64 (__fastcall *v12)(_QWORD, _QWORD, _QWORD); // r14
  __int64 (__fastcall *v13)(_QWORD, _QWORD); // r13
  int v14; // ebx
  __int64 v15; // rsi
  int v16; // r12d
  char v17; // dl
  int v19; // [rsp+8h] [rbp-48h]
  int v20; // [rsp+Ch] [rbp-44h]
  __int64 (__fastcall *v21)(_QWORD, _QWORD); // [rsp+10h] [rbp-40h]
  __int64 (__fastcall *v22)(_QWORD, _QWORD); // [rsp+18h] [rbp-38h]

  v9 = qword_4F08028;
  v10 = a7;
  qword_4F08028 = a3;
  v11 = qword_4F08040;
  v12 = qword_4F08038;
  qword_4F08040 = a1;
  v22 = v9;
  qword_4F08038 = a2;
  v13 = qword_4F08030;
  v14 = dword_4F08018;
  v15 = a8;
  v21 = qword_4F08020;
  qword_4F08030 = a5;
  qword_4F08020 = a4;
  v20 = dword_4F08014;
  dword_4F08018 = a6;
  v19 = dword_4F08010;
  v16 = dword_4D03B64;
  if ( a3 )
  {
    v15 = a8;
    v10 = (__int64 *)((__int64 (__fastcall *)(__int64 *))a3)(a7);
  }
  v17 = *((_BYTE *)v10 - 8);
  dword_4F08010 = (v17 & 2) != 0;
  dword_4F08014 = v17 & 1;
  sub_7506E0(v10, v15);
  qword_4F08040 = v11;
  qword_4F08038 = v12;
  qword_4F08028 = v22;
  qword_4F08030 = v13;
  qword_4F08020 = v21;
  dword_4F08018 = v14;
  dword_4F08014 = v20;
  dword_4F08010 = v19;
  dword_4D03B64 = v16;
  return &dword_4D03B64;
}
