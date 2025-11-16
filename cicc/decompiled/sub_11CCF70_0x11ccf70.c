// Function: sub_11CCF70
// Address: 0x11ccf70
//
__int64 __fastcall sub_11CCF70(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8)
{
  __int64 v12; // rax
  unsigned __int8 *v13; // rdx
  unsigned __int8 *v14; // r12
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // r13
  unsigned __int8 *v18; // rax
  __int64 v20; // [rsp+18h] [rbp-88h]
  unsigned __int64 v21; // [rsp+20h] [rbp-80h]
  _QWORD v22[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v23[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v24; // [rsp+60h] [rbp-40h]

  v20 = sub_AA4B30(*(_QWORD *)(a6 + 48));
  v12 = sub_11CCEE0(v20, a8, a3, 0, *(__int64 **)(a1 + 8), *(_QWORD *)(a1 + 8), *(_QWORD *)(a2 + 8));
  v14 = v13;
  v21 = v12;
  sub_11C9500(v20, a4, a5, a8);
  v22[0] = a1;
  v24 = 261;
  v23[0] = a4;
  v23[1] = a5;
  v22[1] = a2;
  v15 = sub_921880((unsigned int **)a6, v21, (int)v14, (int)v22, 2, (__int64)v23, 0);
  v16 = *(__int64 **)(a6 + 72);
  v17 = v15;
  *(_QWORD *)(v15 + 72) = sub_A7B980(a7, v16, -1, 67);
  v18 = sub_BD3990(v14, (__int64)v16);
  if ( !*v18 )
    *(_WORD *)(v17 + 2) = *(_WORD *)(v17 + 2) & 0xF003 | (4 * ((*((_WORD *)v18 + 1) >> 4) & 0x3FF));
  return v17;
}
