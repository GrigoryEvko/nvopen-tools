// Function: sub_1648AB0
// Address: 0x1648ab0
//
_QWORD *__fastcall sub_1648AB0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v6; // r15
  _QWORD *v7; // r12
  _QWORD *v9; // rax

  v3 = 24LL * a2;
  if ( a3 )
  {
    v4 = a3 + 8;
    v6 = sub_22077B0(v3 + v4 + a1);
    v7 = (_QWORD *)(v6 + v4 + v3);
    *((_DWORD *)v7 + 5) = *((_DWORD *)v7 + 5) & 0x30000000 | a2 & 0xFFFFFFF | 0x80000000;
    sub_16485A0((_QWORD *)(v6 + v4), v7);
    *(_QWORD *)(v6 + a3) = a3;
  }
  else
  {
    v9 = (_QWORD *)sub_22077B0(v3 + a1);
    v7 = &v9[(unsigned __int64)v3 / 8];
    *((_DWORD *)v7 + 5) = *((_DWORD *)v7 + 5) & 0x30000000 | a2 & 0xFFFFFFF;
    sub_16485A0(v9, v7);
  }
  return v7;
}
