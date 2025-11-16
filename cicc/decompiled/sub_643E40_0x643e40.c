// Function: sub_643E40
// Address: 0x643e40
//
__int64 __fastcall sub_643E40(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  char v5; // dl

  result = qword_4CFDE70;
  if ( qword_4CFDE70 )
    qword_4CFDE70 = *(_QWORD *)qword_4CFDE70;
  else
    result = sub_823970(24);
  v5 = *(_BYTE *)(result + 16);
  *(_QWORD *)(result + 8) = a1;
  *(_BYTE *)(result + 16) = a3 & 1 | v5 & 0xFE;
  *(_QWORD *)result = *(_QWORD *)(a2 + 432);
  *(_QWORD *)(a2 + 432) = result;
  return result;
}
