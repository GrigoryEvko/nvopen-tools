// Function: sub_8972D0
// Address: 0x8972d0
//
_QWORD *__fastcall sub_8972D0(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r12
  __int64 v3; // r13
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  _QWORD v7[60]; // [rsp+0h] [rbp-400h] BYREF
  _BYTE v8[192]; // [rsp+1E0h] [rbp-220h] BYREF
  _QWORD *v9; // [rsp+2A0h] [rbp-160h]
  _QWORD *v10; // [rsp+330h] [rbp-D0h]

  v1 = sub_88D660();
  v2 = sub_87EBB0(0x13u, v1, &dword_4F077C8);
  v3 = v2[11];
  *((_DWORD *)v2 + 10) = *(_DWORD *)(qword_4F07288 + 24);
  v4 = (_QWORD *)sub_878CA0();
  *(_QWORD *)(v3 + 32) = v4;
  v5 = v4;
  *(_QWORD *)(v3 + 256) = v4;
  *v4 = a1;
  *(_DWORD *)(v3 + 264) = *(_DWORD *)(v3 + 264) & 0xFFBFFF00 | 0x400009;
  memset(v7, 0, 0x1D8u);
  v7[19] = v7;
  v7[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v7[22]) |= 1u;
  sub_891F00((__int64)v8, (__int64)v7);
  v10 = sub_727340();
  v9 = v5;
  sub_896F00((__int64)v8, (__int64)v2, v3, 0, 0);
  return v2;
}
