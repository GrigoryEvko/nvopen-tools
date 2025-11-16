// Function: sub_760370
// Address: 0x760370
//
__int64 __fastcall sub_760370(_QWORD *a1, unsigned __int8 a2)
{
  __int64 (__fastcall *v2)(_QWORD, _QWORD); // r8
  __int64 (__fastcall *v3)(_QWORD, _QWORD); // r10
  int v4; // r13d
  __int64 (__fastcall *v5)(_QWORD, _QWORD, _QWORD); // r9
  int v6; // r14d
  int v7; // r12d
  int v8; // r15d
  __int64 result; // rax
  __int64 (__fastcall *v10)(_QWORD, _QWORD); // [rsp+0h] [rbp-60h]
  __int64 (__fastcall *v11)(_QWORD, _QWORD, _QWORD); // [rsp+8h] [rbp-58h]
  __int64 (__fastcall *v12)(_QWORD, _QWORD); // [rsp+10h] [rbp-50h]
  __int64 (__fastcall *v13)(_QWORD, _QWORD); // [rsp+18h] [rbp-48h]
  __int64 (__fastcall *v14)(_QWORD, _QWORD); // [rsp+20h] [rbp-40h]

  v2 = qword_4F08030;
  v3 = qword_4F08040;
  qword_4F08040 = 0;
  v4 = dword_4D03B64;
  v5 = qword_4F08038;
  qword_4F08038 = 0;
  qword_4F08030 = (__int64 (__fastcall *)(_QWORD, _QWORD))sub_7607F0;
  v10 = v3;
  v6 = dword_4F08010;
  v7 = dword_4F08018;
  v11 = v5;
  v12 = v2;
  v8 = dword_4F08014;
  v13 = qword_4F08028;
  v14 = qword_4F08020;
  dword_4F08010 = (*(_BYTE *)(a1 - 1) & 2) != 0;
  qword_4F08028 = 0;
  qword_4F08020 = 0;
  dword_4F08018 = 0;
  sub_75C0C0(a1, a2);
  dword_4F08014 = v8;
  dword_4F08010 = v6;
  dword_4D03B64 = v4;
  qword_4F08040 = v10;
  qword_4F08038 = v11;
  result = dword_4F06C5C;
  qword_4F08030 = v12;
  qword_4F08028 = v13;
  qword_4F08020 = v14;
  dword_4F08018 = v7;
  if ( dword_4F06C5C )
    return sub_75B260((__int64)a1, a2);
  return result;
}
