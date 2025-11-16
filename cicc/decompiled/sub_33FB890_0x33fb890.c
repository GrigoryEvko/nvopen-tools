// Function: sub_33FB890
// Address: 0x33fb890
//
unsigned __int8 *__fastcall sub_33FB890(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __m128i a6)
{
  __int64 v6; // r9
  unsigned __int8 *v8; // r12
  __int64 v10; // rax
  __int64 v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  int v15; // [rsp+18h] [rbp-38h]

  v6 = a1;
  v8 = (unsigned __int8 *)a4;
  v10 = *(_QWORD *)(a4 + 48) + 16LL * a5;
  if ( (_WORD)a2 != *(_WORD *)v10 || !(_WORD)a2 && *(_QWORD *)(v10 + 8) != a3 )
  {
    v12 = *(_QWORD *)(a4 + 80);
    v14 = v12;
    if ( v12 )
    {
      v13 = a4;
      sub_B96E90((__int64)&v14, v12, 1);
      v6 = a1;
      a4 = v13;
    }
    v15 = *(_DWORD *)(a4 + 72);
    v8 = sub_33FAF80(v6, 234, (__int64)&v14, a2, a3, v6, a6);
    if ( v14 )
      sub_B91220((__int64)&v14, v14);
  }
  return v8;
}
