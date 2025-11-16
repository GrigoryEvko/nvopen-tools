// Function: sub_87F550
// Address: 0x87f550
//
__int64 sub_87F550()
{
  _QWORD *v1; // rax
  __int64 v2; // rbx
  __int64 *v3; // r12
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rax

  if ( qword_4F5FFC8 )
    return qword_4F5FFC8;
  v1 = sub_87EBB0(0x13u, 0, &dword_4F077C8);
  *((_BYTE *)v1 + 81) |= 0x60u;
  v2 = v1[11];
  v3 = v1;
  v4 = sub_727340();
  v4[21] = v2;
  v5 = v4;
  sub_877D80((__int64)v4, v3);
  *((_BYTE *)v5 + 120) = 8;
  *(_BYTE *)(v2 + 160) |= 4u;
  *(_QWORD *)(v2 + 104) = v5;
  *(_BYTE *)(v2 + 264) = 9;
  v6 = sub_878CA0();
  qword_4F5FFC8 = (__int64)v3;
  *(_QWORD *)(v2 + 32) = v6;
  return (__int64)v3;
}
