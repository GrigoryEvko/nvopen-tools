// Function: sub_32468A0
// Address: 0x32468a0
//
__int64 __fastcall sub_32468A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0x2000000000LL;
  *(_QWORD *)(a1 + 24) = a2;
  result = *(unsigned __int8 *)(a3 + 976);
  *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 60) = result;
  return result;
}
