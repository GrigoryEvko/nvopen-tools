// Function: sub_ED3B20
// Address: 0xed3b20
//
__int64 *__fastcall sub_ED3B20(__int64 *a1, __int64 a2, char *a3, __int64 a4)
{
  _QWORD *v6; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v9)(__m128i **, const __m128i **, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v10)(__int64, char ***, _QWORD *); // [rsp+18h] [rbp-28h]

  v9 = 0;
  v6 = (_QWORD *)sub_22077B0(24);
  if ( v6 )
  {
    v6[1] = 0;
    v6[2] = a2;
    *v6 = &sub_ED3FC0;
  }
  v8[0] = v6;
  v10 = sub_ED02A0;
  v9 = sub_ED0500;
  sub_ED34E0(a1, a3, a4, (__int64)v8);
  if ( v9 )
    v9((__m128i **)v8, (const __m128i **)v8, 3);
  return a1;
}
