// Function: sub_67D470
// Address: 0x67d470
//
__int64 __fastcall sub_67D470(unsigned int a1, unsigned __int8 a2, unsigned int *a3)
{
  unsigned __int64 v4; // rbx
  __int64 result; // rax

  v4 = (*((unsigned __int16 *)a3 + 2) + 1LL) * (*a3 + 1LL) * a1 * ((unsigned __int64)a2 + 1) % 0x3D7;
  result = sub_823970(32);
  *(_DWORD *)(result + 8) = a1;
  *(_BYTE *)(result + 12) = a2;
  *(_QWORD *)(result + 16) = *(_QWORD *)a3;
  *(_QWORD *)result = qword_4CFDEC0[(int)v4];
  *(_QWORD *)(result + 24) = 0xFFFFFFFFLL;
  qword_4CFDEC0[(int)v4] = result;
  return result;
}
