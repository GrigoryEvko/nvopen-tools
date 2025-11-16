// Function: sub_3813100
// Address: 0x3813100
//
__int64 *__fastcall sub_3813100(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int128 *v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // r11
  __int64 v13; // rdx
  __int64 *v14; // r14
  __int128 v16; // [rsp-20h] [rbp-60h]
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  int v18; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 339 )
  {
    v5 = v4[5];
    v6 = v4[6];
  }
  else
  {
    v5 = v4[10];
    v6 = v4[11];
  }
  v7 = *(_QWORD *)(a2 + 80);
  v17 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v17, v7, 1);
  v18 = *(_DWORD *)(a2 + 72);
  v9 = sub_380F170(a1, v5, v6);
  v10 = *(__int128 **)(a2 + 40);
  v11 = v8;
  if ( *(_DWORD *)(a2 + 24) == 339 )
    v12 = (__int64 *)(v10 + 5);
  else
    v12 = (__int64 *)v10 + 5;
  v13 = *(_QWORD *)(v9 + 48) + 16LL * (unsigned int)v8;
  *((_QWORD *)&v16 + 1) = v11;
  *(_QWORD *)&v16 = v9;
  v14 = sub_33F34C0(
          *(__int64 **)(a1 + 8),
          339,
          (__int64)&v17,
          *(_WORD *)v13,
          *(_QWORD *)(v13 + 8),
          *(const __m128i **)(a2 + 112),
          *v10,
          v16,
          *v12,
          v12[1]);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v14;
}
