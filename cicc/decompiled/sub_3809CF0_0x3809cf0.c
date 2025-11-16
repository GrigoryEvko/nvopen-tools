// Function: sub_3809CF0
// Address: 0x3809cf0
//
__int64 *__fastcall sub_3809CF0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  unsigned __int64 v4; // r8
  __int64 v5; // r9
  unsigned int *v6; // rdx
  __int64 v7; // rsi
  unsigned __int16 v8; // r15
  __int64 v9; // r13
  __int128 v10; // rax
  __int128 *v11; // rsi
  __int64 *v12; // r8
  __int64 *v13; // r12
  __int64 v15; // [rsp+0h] [rbp-50h]
  unsigned __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h] BYREF
  int v18; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 339 )
  {
    v4 = v3[5];
    v5 = v3[6];
    v6 = (unsigned int *)(v3 + 5);
  }
  else
  {
    v4 = v3[10];
    v5 = v3[11];
    v6 = (unsigned int *)(v3 + 10);
  }
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2]);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2] + 8);
  v17 = v7;
  if ( v7 )
  {
    v15 = v5;
    v16 = v4;
    sub_B96E90((__int64)&v17, v7, 1);
    v5 = v15;
    v4 = v16;
  }
  v18 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v10 = sub_3805E70(a1, v4, v5);
  v11 = *(__int128 **)(a2 + 40);
  if ( *(_DWORD *)(a2 + 24) == 339 )
    v12 = (__int64 *)(v11 + 5);
  else
    v12 = (__int64 *)v11 + 5;
  v13 = sub_33F34C0(
          *(__int64 **)(a1 + 8),
          339,
          (__int64)&v17,
          v8,
          v9,
          *(const __m128i **)(a2 + 112),
          *v11,
          v10,
          *v12,
          v12[1]);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v13;
}
