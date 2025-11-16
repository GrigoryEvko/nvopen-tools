// Function: sub_15C9730
// Address: 0x15c9730
//
__int64 __fastcall sub_15C9730(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _QWORD v6[4]; // [rsp+0h] [rbp-50h] BYREF
  int v7; // [rsp+20h] [rbp-30h]
  __int64 v8; // [rsp+28h] [rbp-28h]

  *(_QWORD *)a1 = a1 + 16;
  if ( a2 )
  {
    sub_15C7EA0((__int64 *)a1, a2, (__int64)&a2[a3]);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  v6[0] = &unk_49EFBE0;
  v8 = a1 + 32;
  v7 = 1;
  memset(&v6[1], 0, 24);
  sub_154E060(a4, (__int64)v6, 0, 0);
  return sub_16E7BC0(v6);
}
