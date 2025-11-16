// Function: sub_30477B0
// Address: 0x30477b0
//
__int64 __fastcall sub_30477B0(__int64 a1, __int64 a2, unsigned int a3, int a4)
{
  __int64 v7; // rsi
  unsigned __int16 *v8; // rbx
  unsigned __int16 v9; // r8
  int v10; // r9d
  __int64 v11; // r13
  __int128 v13; // [rsp-10h] [rbp-40h]
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  int v15; // [rsp+8h] [rbp-28h]

  v7 = *(_QWORD *)(a2 + 80);
  v14 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v14, v7, 1);
  v8 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v15 = *(_DWORD *)(a2 + 72);
  v9 = *v8;
  if ( *v8 == 6 )
  {
    v10 = 1561;
  }
  else
  {
    if ( v9 != 7 )
      BUG();
    v10 = 1564;
  }
  *((_QWORD *)&v13 + 1) = 1;
  *(_QWORD *)&v13 = *(_QWORD *)(a2 + 40);
  v11 = sub_33E6B00(a4, v10, (unsigned int)&v14, v9, 0, v10, v9, v13);
  *(_DWORD *)(v11 + 72) = *(_DWORD *)(a2 + 72);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
  return v11;
}
