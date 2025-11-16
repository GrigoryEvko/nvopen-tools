// Function: sub_B32030
// Address: 0xb32030
//
__int64 __fastcall sub_B32030(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v3; // esi
  int v4; // esi

  sub_B31FB0(a1, a2);
  *(_BYTE *)(a1 + 80) = *(_BYTE *)(a2 + 80) & 2 | *(_BYTE *)(a1 + 80) & 0xFD;
  result = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a1 + 72) = result;
  v3 = *(unsigned __int16 *)(a2 + 34);
  LOWORD(v3) = (unsigned __int16)v3 >> 1;
  v4 = (v3 >> 6) & 7;
  if ( v4 )
    return sub_B30310(a1, v4 - 1);
  return result;
}
