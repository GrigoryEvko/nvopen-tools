// Function: sub_211EDB0
// Address: 0x211edb0
//
__int64 *__fastcall sub_211EDB0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r13
  __int64 v10; // rcx
  const void **v11; // r15
  unsigned int v12; // r12d
  __int64 *v13; // r12
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  int v17; // [rsp+18h] [rbp-58h]
  __int128 v18; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  int v20; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v17 = 0;
  DWORD2(v18) = 0;
  v16 = 0;
  v7 = *(_QWORD *)(v6 + 48);
  *(_QWORD *)&v18 = 0;
  sub_2016B80(a1, *(_QWORD *)(v6 + 40), v7, &v16, &v18);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = *(_QWORD *)(a2 + 32);
  v11 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v12 = **(unsigned __int8 **)(a2 + 40);
  v19 = v8;
  if ( v8 )
  {
    v15 = v10;
    sub_1623A60((__int64)&v19, v8, 2);
    v10 = v15;
  }
  v20 = *(_DWORD *)(a2 + 64);
  v13 = sub_1D332F0(v9, 101, (__int64)&v19, v12, v11, 0, a3, a4, a5, *(_QWORD *)v10, *(_QWORD *)(v10 + 8), v18);
  if ( v19 )
    sub_161E7C0((__int64)&v19, v19);
  return v13;
}
