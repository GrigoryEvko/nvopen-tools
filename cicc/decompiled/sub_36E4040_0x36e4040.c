// Function: sub_36E4040
// Address: 0x36e4040
//
__int64 __fastcall sub_36E4040(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r14
  int v7; // r13d
  __int64 v8; // rax
  _QWORD *v9; // rsi
  unsigned __int8 *v10; // rax
  int v11; // edx
  _WORD *v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // r9
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v21; // [rsp+8h] [rbp-78h]
  int v22; // [rsp+10h] [rbp-70h]
  _WORD *v23; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-68h]
  _QWORD *v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+20h] [rbp-60h] BYREF
  int v27; // [rsp+28h] [rbp-58h]
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  int v29; // [rsp+38h] [rbp-48h]
  __int64 v30; // [rsp+40h] [rbp-40h]
  int v31; // [rsp+48h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *v4;
  v7 = *((_DWORD *)v4 + 2);
  v28 = v5;
  if ( v5 )
  {
    sub_B96E90((__int64)&v28, v5, 1);
    v4 = *(__int64 **)(a2 + 40);
  }
  v29 = *(_DWORD *)(a2 + 72);
  v8 = *(_QWORD *)(v4[10] + 96);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v9, (__int64)&v28, 7, 0, 1u, a3, 0);
  if ( v28 )
  {
    v22 = v11;
    v24 = v10;
    sub_B91220((__int64)&v28, v28);
    v11 = v22;
    v10 = v24;
  }
  v12 = *(_WORD **)(a2 + 48);
  if ( *v12 != 7 )
    sub_C64ED0("Unsupported overloaded declaration of llvm.nvvm.read.sreg intrinsic", 1u);
  v13 = *(_QWORD *)(a2 + 80);
  v30 = v6;
  v14 = *(_QWORD **)(a1 + 64);
  v15 = *(unsigned int *)(a2 + 68);
  v31 = v7;
  v28 = (__int64)v10;
  v29 = v11;
  v26 = v13;
  if ( v13 )
  {
    v21 = v15;
    v23 = v12;
    v25 = v14;
    sub_B96E90((__int64)&v26, v13, 1);
    v15 = v21;
    v12 = v23;
    v14 = v25;
  }
  v27 = *(_DWORD *)(a2 + 72);
  v16 = sub_33E66D0(v14, 3457, (__int64)&v26, (unsigned __int64)v12, v15, (__int64)v14, (unsigned __int64 *)&v28, 2);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v16, v17, v18, v19);
  sub_3421DB0(v16);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return 1;
}
