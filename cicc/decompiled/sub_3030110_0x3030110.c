// Function: sub_3030110
// Address: 0x3030110
//
__int64 __fastcall sub_3030110(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int16 *v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // r13d
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  int v13; // [rsp+8h] [rbp-38h]

  v6 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v12 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v12, v7, 1);
  v13 = *(_DWORD *)(a2 + 72);
  v10 = sub_33FE730(a4, &v12, v8, v9, 0, 0.0);
  if ( v12 )
    sub_B91220((__int64)&v12, v12);
  return v10;
}
