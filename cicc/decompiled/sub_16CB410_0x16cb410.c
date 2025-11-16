// Function: sub_16CB410
// Address: 0x16cb410
//
__int64 *__fastcall sub_16CB410(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rax
  bool v4; // zf
  _BYTE *v5; // rsi
  unsigned __int64 v6; // rdx

  v3 = sub_2241A80(a2, 48, -1);
  v4 = *(_BYTE *)(*a2 + v3) == 46;
  *a1 = (__int64)(a1 + 2);
  v5 = (_BYTE *)*a2;
  v6 = v3 + v4 + 1;
  if ( v6 > a2[1] )
    v6 = a2[1];
  sub_16CB2B0(a1, v5, (__int64)&v5[v6]);
  return a1;
}
