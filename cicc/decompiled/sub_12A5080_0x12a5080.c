// Function: sub_12A5080
// Address: 0x12a5080
//
__int64 __fastcall sub_12A5080(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // r13
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdi

  v2 = (_DWORD *)(a2 + 64);
  sub_12A4F30(a1, (_DWORD *)(a2 + 64));
  v4 = sub_1297B70((_QWORD **)(a1[4] + 8LL), a2, 0);
  sub_1298A30((__int64)a1, v4, v2);
  if ( !dword_4D04658 )
  {
    v4 = (__int64)(a1 + 6);
    sub_129F180(*(_QWORD **)(a1[4] + 384LL), a1 + 6);
  }
  v7 = a1[46];
  a1[46] = 0;
  return sub_15F20C0(v7, v4, v5, v6);
}
