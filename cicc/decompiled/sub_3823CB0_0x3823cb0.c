// Function: sub_3823CB0
// Address: 0x3823cb0
//
void __fastcall sub_3823CB0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // r9
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r10
  __int64 *v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // r9
  unsigned int v21; // edx
  unsigned __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v23 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v23, v9, 1);
  v10 = *a1;
  v11 = a1[1];
  v24 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = *(_QWORD *)(*(_QWORD *)(v12 + 80) + 96LL);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = sub_34696C0(
          v10,
          *(_DWORD *)(a2 + 24),
          (__int64)&v23,
          *(_QWORD *)v12,
          *(_QWORD *)(v12 + 8),
          (unsigned int)v14,
          a5,
          *(_OWORD *)(v12 + 40),
          v11);
  v17 = v16;
  if ( !v15 )
  {
    v18 = *(__int64 **)(a2 + 40);
    v19 = *(_QWORD *)(v18[10] + 96);
    v20 = *(_QWORD **)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) > 0x40u )
      v20 = (_QWORD *)*v20;
    v22 = v17;
    v15 = sub_3813E70(a2, *v18, v18[1], v18[5], v18[6], (unsigned int)v20, a5, *a1, a1[1], 0);
    v17 = v21 | v22 & 0xFFFFFFFF00000000LL;
  }
  sub_375BC20(a1, (__int64)v15, v17, a3, a4, a5);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
}
