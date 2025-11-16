// Function: sub_23DF780
// Address: 0x23df780
//
__int64 __fastcall sub_23DF780(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        char a4,
        __int64 a5,
        __int16 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  __int64 result; // rax
  __int64 v14; // r12

  v10 = a3;
  *(_BYTE *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 48) = a7;
  *(_WORD *)(a1 + 40) = a6;
  *(_QWORD *)(a1 + 56) = a8;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 64) = a9;
  v11 = sub_B43CC0(a2);
  *(_QWORD *)(a1 + 24) = (sub_9208B0(v11, a5) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  result = v12;
  *(_BYTE *)(a1 + 32) = v12;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v14 = *(_QWORD *)(a2 - 8);
  }
  else
  {
    result = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v14 = a2 - result;
  }
  *(_QWORD *)a1 = 32 * v10 + v14;
  return result;
}
