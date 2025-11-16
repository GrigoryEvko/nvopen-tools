// Function: sub_11CC8D0
// Address: 0x11cc8d0
//
__int64 __fastcall sub_11CC8D0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 *a7)
{
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int8 *v13; // rdx
  unsigned __int8 *v14; // r13
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // r12
  unsigned __int8 *v18; // rax
  __int64 v20; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v21[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  v20 = a1;
  v11 = sub_AA4B30(*(_QWORD *)(a5 + 48));
  v12 = sub_11CC840(v11, a7, a2, 0, *(__int64 **)(v20 + 8), *(_QWORD *)(v20 + 8));
  v14 = v13;
  v21[1] = a4;
  v22 = 261;
  v21[0] = a3;
  v15 = sub_921880((unsigned int **)a5, v12, (int)v13, (int)&v20, 1, (__int64)v21, 0);
  v16 = *(__int64 **)(a5 + 72);
  v17 = v15;
  *(_QWORD *)(v15 + 72) = sub_A7B980(a6, v16, -1, 67);
  v18 = sub_BD3990(v14, (__int64)v16);
  if ( !*v18 )
    *(_WORD *)(v17 + 2) = *(_WORD *)(v17 + 2) & 0xF003 | (4 * ((*((_WORD *)v18 + 1) >> 4) & 0x3FF));
  return v17;
}
