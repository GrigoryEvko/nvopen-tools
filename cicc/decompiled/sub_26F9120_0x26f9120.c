// Function: sub_26F9120
// Address: 0x26f9120
//
__int64 __fastcall sub_26F9120(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned int a8)
{
  __int64 v14; // rsi
  unsigned __int8 *v15; // r14
  unsigned __int8 *v16; // r13
  __int64 v17; // r14
  __int64 v18; // rdi
  int v19; // r12d
  __int64 v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // r12
  __int64 v27; // rax
  __m128i v28; // [rsp+0h] [rbp-60h]
  unsigned int v29; // [rsp+14h] [rbp-4Ch]
  __int64 v31[8]; // [rsp+20h] [rbp-40h] BYREF

  v29 = a8;
  v28 = _mm_loadu_si128((const __m128i *)&a7);
  if ( !(unsigned __int8)sub_26F8EA0(*a1) )
    return sub_ACD640(a6, v29, 0);
  v14 = a1[12];
  v15 = sub_26F9080(*a1, v14, a2, a3, a4, a5, (unsigned __int8 *)v28.m128i_i64[0], v28.m128i_u64[1]);
  v16 = sub_BD3990(v15, v14);
  v17 = sub_AD4C50((unsigned __int64)v15, (__int64 **)a6, 0);
  if ( (v16[7] & 0x20) == 0 || !sub_B91C10((__int64)v16, 21) )
  {
    v18 = a1[11];
    v19 = *(_DWORD *)(a6 + 8) >> 8;
    if ( *(_DWORD *)(v18 + 8) >> 8 == v19 )
    {
      v25 = sub_ACD640(v18, -1, 0);
      v26 = sub_B98A20(v25, -1);
      v27 = sub_ACD640(a1[11], -1, 0);
      v23 = sub_B98A20(v27, -1);
      v31[0] = (__int64)v26;
    }
    else
    {
      v20 = sub_ACD640(v18, 0, 0);
      v21 = sub_B98A20(v20, 0);
      v22 = sub_ACD640(a1[11], 1LL << v19, 0);
      v23 = sub_B98A20(v22, 1LL << v19);
      v31[0] = (__int64)v21;
    }
    v31[1] = (__int64)v23;
    v24 = sub_B9C770(*(__int64 **)*a1, v31, (__int64 *)2, 0, 1);
    sub_B99110((__int64)v16, 21, v24);
  }
  return v17;
}
