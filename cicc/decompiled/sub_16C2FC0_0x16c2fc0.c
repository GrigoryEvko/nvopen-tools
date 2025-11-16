// Function: sub_16C2FC0
// Address: 0x16c2fc0
//
_QWORD *__fastcall sub_16C2FC0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v6; // rdx

  v2 = a2[1];
  v3 = a2[2] - v2;
  v4 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 16LL))(a2);
  *a1 = v2;
  a1[1] = v3;
  a1[2] = v4;
  a1[3] = v6;
  return a1;
}
