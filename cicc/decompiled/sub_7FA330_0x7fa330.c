// Function: sub_7FA330
// Address: 0x7fa330
//
__int64 sub_7FA330()
{
  _QWORD *v0; // rax
  __int64 v1; // rbx
  _QWORD *v2; // rdx
  __int64 result; // rax

  v0 = sub_725A70(0);
  *((_BYTE *)v0 + 51) |= 0x20u;
  v1 = (__int64)v0;
  v0[10] = sub_7F9300();
  sub_732DB0(v1, qword_4F06BC0, 1);
  v2 = qword_4D03F68;
  result = *(_QWORD *)(v1 + 80);
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_DWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)(result + 48) = 0;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = -1;
  *(_QWORD *)(result + 80) = v2[6];
  v2[6] = v1;
  v2[5] = v1;
  return result;
}
