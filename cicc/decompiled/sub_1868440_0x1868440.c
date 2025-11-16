// Function: sub_1868440
// Address: 0x1868440
//
__int64 __fastcall sub_1868440(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r9
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 v21; // [rsp+10h] [rbp-30h] BYREF
  __int64 v22[5]; // [rsp+18h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_1636800(a1, a2) )
    return 0;
  v11 = sub_1632FA0((__int64)a2);
  v12 = *(__int64 **)(a1 + 8);
  v13 = (__int64 *)v11;
  v14 = *v12;
  v15 = v12[1];
  if ( v14 == v15 )
LABEL_8:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9B6E8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_8;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9B6E8);
  v20 = a1;
  v21 = a1;
  v22[0] = a1;
  return sub_1866840(
           a2,
           v13,
           (_QWORD *)(v16 + 360),
           (__int64 (__fastcall *)(__int64, _QWORD *))sub_185B170,
           (__int64)&v21,
           a3,
           a4,
           a5,
           a6,
           v18,
           v19,
           a9,
           a10,
           v17,
           (__int64 (__fastcall *)(__int64, _QWORD *))sub_185AFD0,
           (__int64)v22,
           (__int64 (__fastcall *)(__int64, _QWORD *))sub_185AF90,
           (__int64)&v20);
}
