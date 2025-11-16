// Function: sub_75AFC0
// Address: 0x75afc0
//
__int64 __fastcall sub_75AFC0(
        int a1,
        __int64 (__fastcall *a2)(_QWORD, _QWORD),
        __int64 (__fastcall *a3)(_QWORD, _QWORD, _QWORD),
        __int64 (__fastcall *a4)(_QWORD, _QWORD),
        __int64 (__fastcall *a5)(_QWORD, _QWORD),
        __int64 (__fastcall *a6)(_QWORD, _QWORD),
        int a7)
{
  __int64 (__fastcall *v8)(_QWORD, _QWORD); // rax
  __int64 (__fastcall *v9)(_QWORD, _QWORD); // r15
  __int64 (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // r14
  __int64 (__fastcall *v11)(_QWORD, _QWORD); // r13
  __int64 *v12; // r12
  __int64 v13; // rax
  unsigned int v15; // [rsp+0h] [rbp-50h]
  int v16; // [rsp+4h] [rbp-4Ch]
  int v17; // [rsp+8h] [rbp-48h]
  int v18; // [rsp+Ch] [rbp-44h]
  __int64 (__fastcall *v19)(_QWORD, _QWORD); // [rsp+10h] [rbp-40h]
  __int64 (__fastcall *v20)(_QWORD, _QWORD); // [rsp+18h] [rbp-38h]

  v8 = qword_4F08028;
  v9 = qword_4F08040;
  qword_4F08028 = a4;
  v10 = qword_4F08038;
  v11 = qword_4F08030;
  qword_4F08038 = a3;
  v20 = v8;
  qword_4F08040 = a2;
  v19 = qword_4F08020;
  qword_4F08030 = a6;
  v18 = dword_4F08014;
  qword_4F08020 = a5;
  v17 = dword_4F08010;
  dword_4F08014 = 0;
  v16 = dword_4D03B64;
  v15 = dword_4F08018;
  dword_4F08018 = a7;
  v12 = *(__int64 **)(unk_4F072B0 + 8LL * a1);
  dword_4D03B64 = ((*((_BYTE *)v12 - 8) >> 2) ^ 1) & 1;
  dword_4F08010 = (*(_BYTE *)(v12 - 1) & 2) != 0;
  do
  {
    while ( 1 )
    {
      sub_7506E0(v12, 23);
      if ( !a4 )
        break;
      v13 = a4(*v12, 23);
      *v12 = v13;
      v12 = (__int64 *)v13;
      if ( !v13 )
        goto LABEL_5;
    }
    v12 = (__int64 *)*v12;
  }
  while ( v12 );
LABEL_5:
  qword_4F08040 = v9;
  qword_4F08038 = v10;
  qword_4F08028 = v20;
  qword_4F08030 = v11;
  qword_4F08020 = v19;
  dword_4F08014 = v18;
  dword_4F08010 = v17;
  dword_4D03B64 = v16;
  dword_4F08018 = v15;
  return v15;
}
