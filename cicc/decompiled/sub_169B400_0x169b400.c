// Function: sub_169B400
// Address: 0x169b400
//
__int64 __fastcall sub_169B400(__int64 a1, char a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  int v4; // ecx
  __int64 result; // rax

  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF8 | 2;
  sub_1698870(a1);
  v2 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a1 + 18) & 0xF7 | (8 * (a2 & 1));
  *(_WORD *)(a1 + 16) = *(_WORD *)(v2 + 2);
  v3 = sub_1698470(a1);
  v4 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  result = 1LL << ((unsigned __int8)v4 - 1);
  *(_QWORD *)(v3 + 8LL * (((unsigned int)(v4 + 63) >> 6) - 1)) |= result;
  return result;
}
