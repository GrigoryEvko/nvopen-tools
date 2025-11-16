// Function: sub_31B0F60
// Address: 0x31b0f60
//
_QWORD *__fastcall sub_31B0F60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rcx
  _DWORD *v6; // rdx
  __int64 v7; // r8
  __m128i v9; // [rsp+0h] [rbp-90h] BYREF
  __int16 v10; // [rsp+10h] [rbp-80h]
  __int64 v11; // [rsp+18h] [rbp-78h]
  __m128i v12; // [rsp+20h] [rbp-70h]
  __int64 v13; // [rsp+30h] [rbp-60h]
  __int64 v14; // [rsp+38h] [rbp-58h]
  __int64 v15[4]; // [rsp+40h] [rbp-50h] BYREF
  char v16; // [rsp+60h] [rbp-30h]
  char v17; // [rsp+61h] [rbp-2Fh]

  v15[0] = a2;
  sub_31AFE90(&v9, v15, 1, a4);
  v5 = *(unsigned int *)(a3 + 8);
  v6 = *(_DWORD **)a3;
  v7 = *(_QWORD *)(a2 + 24);
  v17 = 1;
  v16 = 3;
  v15[0] = (__int64)"VShuf";
  v12 = v9;
  LOWORD(v13) = v10;
  v14 = v11;
  return sub_318BF10(a2, a2, v6, v5, v7, (__int64)v15, v9.m128i_i64[0], v9.m128i_i64[1], v13);
}
