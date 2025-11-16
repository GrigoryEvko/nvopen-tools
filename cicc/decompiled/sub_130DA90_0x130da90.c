// Function: sub_130DA90
// Address: 0x130da90
//
__int64 __fastcall sub_130DA90(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 (__fastcall **v4)(int, int, int, int, int, int, int); // rax
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  __int64 result; // rax

  v4 = *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(a2 + 8);
  v5 = a3[2] & 0xFFFFFFFFFFFFF000LL;
  v6 = a3[1] & 0xFFFFFFFFFFFFF000LL;
  if ( v4 == &off_49E8020 )
    sub_1341250(0, (_BYTE *)(v6 + v5));
  v7 = a3[2];
  a3[1] = v6;
  *a3 &= ~0x10000uLL;
  result = (v5 + 4096) | v7 & 0xFFF;
  a3[2] = result;
  return result;
}
