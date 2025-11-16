// Function: sub_D89F40
// Address: 0xd89f40
//
_QWORD *__fastcall sub_D89F40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v6)(const __m128i **, const __m128i *, int); // [rsp+10h] [rbp-20h]
  __int64 (__fastcall *v7)(__int64 *); // [rsp+18h] [rbp-18h]

  v5[1] = a3;
  v7 = sub_D85610;
  v6 = sub_D857C0;
  v5[0] = a4;
  sub_D898E0(a1, a3, (__int64)v5);
  if ( v6 )
    v6((const __m128i **)v5, (const __m128i *)v5, 3);
  return a1;
}
