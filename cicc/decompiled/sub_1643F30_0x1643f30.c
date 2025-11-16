// Function: sub_1643F30
// Address: 0x1643f30
//
__int64 __fastcall sub_1643F30(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // rax
  int v4; // edx
  __int64 result; // rax

  v3 = *a2;
  *(_BYTE *)(a1 + 8) = 15;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)a1 = v3;
  *(_QWORD *)(a1 + 16) = a1 + 24;
  LODWORD(v3) = a3;
  v4 = *(unsigned __int8 *)(a1 + 8);
  result = (unsigned int)((_DWORD)v3 << 8);
  *(_DWORD *)(a1 + 12) = 1;
  *(_DWORD *)(a1 + 8) = result | v4;
  return result;
}
