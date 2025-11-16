// Function: sub_72B6D0
// Address: 0x72b6d0
//
_QWORD *__fastcall sub_72B6D0(__int64 a1, int a2)
{
  _QWORD *v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax

  v2 = sub_7259C0(14);
  v3 = v2[21];
  v4 = sub_87F7E0(3, a1);
  *v2 = v4;
  *(_QWORD *)(v4 + 88) = v2;
  *(_DWORD *)(v3 + 28) = -1;
  *(_DWORD *)(v3 + 24) = 1 - ((a2 == 0) - 1);
  sub_8D6090(v2);
  return v2;
}
