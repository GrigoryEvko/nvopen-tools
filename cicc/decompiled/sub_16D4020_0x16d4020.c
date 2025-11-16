// Function: sub_16D4020
// Address: 0x16d4020
//
__int64 __fastcall sub_16D4020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 result; // rax

  if ( !*(_QWORD *)(a1 + 16) )
  {
    result = qword_4FA13E8;
    *(_QWORD *)(a1 + 16) = a2;
    qword_4FA13E8 = a1;
    *(_QWORD *)a1 = result;
    *(_QWORD *)(a1 + 24) = a3;
    *(_QWORD *)(a1 + 32) = a4;
    *(_QWORD *)(a1 + 8) = a5;
    *(_BYTE *)(a1 + 40) = a6;
  }
  return result;
}
