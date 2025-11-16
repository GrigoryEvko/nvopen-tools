// Function: sub_1D45080
// Address: 0x1d45080
//
__int64 *__fastcall sub_1D45080(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // r14
  unsigned int v13; // edx
  unsigned int v14; // r11d
  __int128 v15; // [rsp-10h] [rbp-70h]
  unsigned int v16; // [rsp+8h] [rbp-58h]
  __int64 *v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  int v20; // [rsp+28h] [rbp-38h]

  if ( !(unsigned __int8)sub_1D18C40(a2, 1) )
    return (__int64 *)a3;
  v10 = *(_QWORD *)(a2 + 72);
  v19 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v19, v10, 2);
  v11 = *(_DWORD *)(a2 + 64);
  v12 = a3;
  v20 = v11;
  *((_QWORD *)&v15 + 1) = 1;
  *(_QWORD *)&v15 = a3;
  v18 = sub_1D332F0(a1, 2, (__int64)&v19, 1, 0, 0, a4, a5, a6, a2, 1u, v15);
  v14 = v13;
  if ( v19 )
  {
    v16 = v13;
    sub_161E7C0((__int64)&v19, v19);
    v14 = v16;
  }
  sub_1D44C70((__int64)a1, a2, 1, (__int64)v18, v14);
  sub_1D2DF70(a1, v18, a2, 1, v12, 1);
  return v18;
}
