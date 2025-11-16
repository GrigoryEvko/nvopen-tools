// Function: sub_B44260
// Address: 0xb44260
//
_QWORD *__fastcall sub_B44260(__int64 a1, __int64 a2, int a3, unsigned int a4, __int64 a5, unsigned __int16 a6)
{
  int v6; // r15d
  unsigned int v8; // r14d
  int v9; // ebx
  int v10; // eax
  _QWORD *result; // rax

  v6 = a4 & 0x7FFFFFF;
  v8 = a4 >> 28 << 31;
  v9 = ((a4 >> 27) & 1) << 30;
  sub_BD35F0(a1, a2, (unsigned int)(a3 + 29));
  v10 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  result = (_QWORD *)(v10 & 0x38000000);
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 4) = (unsigned int)result | v8 | v6 | v9;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  if ( a5 )
    return sub_B44240((_QWORD *)a1, *(_QWORD *)(a5 + 16), (unsigned __int64 *)a5, a6);
  return result;
}
