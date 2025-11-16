// Function: sub_39A11A0
// Address: 0x39a11a0
//
__int64 __fastcall sub_39A11A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0x1800000000LL;
  *(_QWORD *)(a1 + 24) = a2;
  v5 = *(_QWORD *)(a3 + 240);
  *(_QWORD *)(a1 + 32) = a4;
  result = *(unsigned __int8 *)(v5 + 356);
  *(_QWORD *)(a1 + 40) = a5;
  *(_DWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 52) = result;
  return result;
}
