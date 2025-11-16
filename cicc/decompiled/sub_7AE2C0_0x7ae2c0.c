// Function: sub_7AE2C0
// Address: 0x7ae2c0
//
__int64 __fastcall sub_7AE2C0(__int16 a1, int a2, _QWORD *a3)
{
  __int64 result; // rax

  result = qword_4F08558;
  if ( qword_4F08558 )
    qword_4F08558 = *(_QWORD *)qword_4F08558;
  else
    result = sub_823970(112);
  *(_QWORD *)(result + 40) = 0;
  *(_QWORD *)result = 0;
  *(_BYTE *)(result + 26) = 0;
  *(_WORD *)(result + 24) = a1;
  *(_DWORD *)(result + 28) = a2;
  *(_DWORD *)(result + 32) = a2;
  *(_QWORD *)(result + 8) = *a3;
  *(_QWORD *)(result + 16) = *a3;
  return result;
}
