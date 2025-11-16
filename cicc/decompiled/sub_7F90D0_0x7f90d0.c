// Function: sub_7F90D0
// Address: 0x7f90d0
//
__int64 __fastcall sub_7F90D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 32) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 48) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a2 + 64) = -1;
  *(_QWORD *)(a2 + 8) = a1;
  *(_DWORD *)(a2 + 16) = 1;
  result = sub_8D46C0(*(_QWORD *)(a1 + 120));
  *(_QWORD *)(a2 + 24) = result;
  return result;
}
