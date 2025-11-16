// Function: sub_88D6D0
// Address: 0x88d6d0
//
_QWORD *__fastcall sub_88D6D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // r14
  _QWORD *v5; // rax
  __m128i *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // rax

  v2 = sub_88D660();
  v3 = sub_87EBB0(0x14u, v2, &dword_4F077C8);
  v4 = (_QWORD *)v3[11];
  *((_DWORD *)v3 + 10) = *(_DWORD *)(qword_4F07288 + 24);
  v5 = (_QWORD *)sub_878CA0();
  v4[4] = v5;
  v4[41] = v5;
  *v5 = a1;
  v6 = sub_725FD0();
  v7 = sub_7259C0(7);
  v6[9].m128i_i64[1] = (__int64)v7;
  v8 = v7[21];
  *(_BYTE *)(v8 + 16) |= 2u;
  v7[20] = sub_72CBE0();
  v9 = sub_724EF0(a2);
  *((_DWORD *)v9 + 9) = 1;
  *(_QWORD *)v8 = v9;
  v4[22] = v6;
  return v3;
}
