// Function: sub_DCAD70
// Address: 0xdcad70
//
_QWORD *__fastcall sub_DCAD70(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  char v5; // bl
  __int64 v6; // rax
  char v7; // dl

  v4 = a1[1];
  v5 = sub_AE5020(v4, a3);
  v6 = sub_9208B0(v4, a3);
  return sub_DCACD0(a1, a2, ((1LL << v5) + ((unsigned __int64)(v6 + 7) >> 3) - 1) >> v5 << v5, v7);
}
