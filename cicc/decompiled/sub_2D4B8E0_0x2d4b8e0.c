// Function: sub_2D4B8E0
// Address: 0x2d4b8e0
//
_BOOL8 __fastcall sub_2D4B8E0(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r10d
  unsigned __int16 v3; // ax
  unsigned __int64 v4; // rdx
  _BOOL8 result; // rax

  v2 = sub_2D44290(a2);
  v3 = *(_WORD *)(a2 + 2);
  _BitScanReverse64(&v4, 1LL << SHIBYTE(v3));
  result = sub_2D4A460(
             a1,
             a2,
             v2,
             63 - ((unsigned __int8)v4 ^ 0x3Fu),
             *(_QWORD *)(a2 - 96),
             *(_QWORD *)(a2 - 32),
             *(_QWORD *)(a2 - 64),
             (v3 >> 2) & 7,
             (v3 >> 5) & 7,
             dword_444C410);
  if ( !result )
    sub_C64ED0("expandAtomicOpToLibcall shouldn't fail for CAS", 1u);
  return result;
}
