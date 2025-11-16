// Function: sub_1ED83F0
// Address: 0x1ed83f0
//
__int64 __fastcall sub_1ED83F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // rax
  __int64 result; // rax
  _QWORD v12[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v13)(__m128i **, const __m128i **, int); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v14)(); // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 272);
  v13 = 0;
  v8 = v7 + 296;
  v9 = (_QWORD *)sub_22077B0(32);
  if ( v9 )
  {
    *v9 = a1;
    v9[1] = v8;
    v9[2] = a3;
    v9[3] = a5;
  }
  v12[0] = v9;
  v14 = sub_1EDCC10;
  v13 = sub_1ED8310;
  sub_1DB5D80(a2, v8, a4, (__int64)v12);
  result = (__int64)v13;
  if ( v13 )
    return v13((__m128i **)v12, (const __m128i **)v12, 3);
  return result;
}
