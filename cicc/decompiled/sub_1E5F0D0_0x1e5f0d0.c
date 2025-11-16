// Function: sub_1E5F0D0
// Address: 0x1e5f0d0
//
_QWORD *__fastcall sub_1E5F0D0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  result = (_QWORD *)sub_22077B0(8);
  *(_QWORD *)(a1 + 56) = a2;
  *result = 0;
  *(_QWORD *)a1 = result;
  *(_QWORD *)(a1 + 16) = result + 1;
  *(_QWORD *)(a1 + 8) = result + 1;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  return result;
}
