// Function: sub_2AC4440
// Address: 0x2ac4440
//
__int64 __fastcall sub_2AC4440(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char v5; // r13
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rax
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(const __m128i **, const __m128i *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(_QWORD *, int *); // [rsp+18h] [rbp-28h]

  v10[0] = a1;
  v12 = sub_2AAAEA0;
  v10[1] = a2;
  v11 = sub_2AA7CC0;
  v5 = sub_2BF1270(v10, a5);
  if ( v11 )
    v11((const __m128i **)v10, (const __m128i *)v10, 3);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(a2 - 32);
  v7 = sub_31A6940(a1[4]);
  v8 = sub_2AC42A0(*a1, *(_QWORD *)(v7 + 16));
  return sub_2ABB2E0(v6, a2, v8, v7, *a1, *(_QWORD *)(a1[6] + 112));
}
