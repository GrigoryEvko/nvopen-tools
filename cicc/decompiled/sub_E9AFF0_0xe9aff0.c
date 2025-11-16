// Function: sub_E9AFF0
// Address: 0xe9aff0
//
__int64 __fastcall sub_E9AFF0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        _DWORD *a7,
        unsigned __int64 a8)
{
  __int8 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r13
  _QWORD *v11; // rax
  unsigned int v13; // [rsp+8h] [rbp-78h]
  __int8 v14; // [rsp+10h] [rbp-70h]
  unsigned __int64 v16; // [rsp+28h] [rbp-58h] BYREF
  __m128i v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+40h] [rbp-40h]
  __int64 v19; // [rsp+48h] [rbp-38h]

  v8 = a4;
  v9 = a1[1];
  v14 = a5;
  v13 = a3;
  v10 = sub_E6C430(v9, a2, a3, a4, a5);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 208LL))(a1, v10, 0);
  v17.m128i_i8[9] = v8;
  v17.m128i_i64[0] = __PAIR64__(a6, v13);
  v17.m128i_i8[8] = v14;
  v18 = a2;
  v19 = v10;
  v16 = a8;
  v11 = (_QWORD *)sub_E9AD20((_QWORD *)(v9 + 1920), &v16);
  return sub_E905B0(v11, &v17, a7);
}
