// Function: sub_1A95AA0
// Address: 0x1a95aa0
//
__int64 __fastcall sub_1A95AA0(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v5; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-98h] BYREF
  __m128i v12; // [rsp+10h] [rbp-90h] BYREF
  int v13; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v14; // [rsp+28h] [rbp-78h]
  int *v15; // [rsp+30h] [rbp-70h]
  int *v16; // [rsp+38h] [rbp-68h]
  __int64 v17; // [rsp+40h] [rbp-60h]
  __int64 v18; // [rsp+48h] [rbp-58h]
  __int64 v19; // [rsp+50h] [rbp-50h]
  __int64 v20; // [rsp+58h] [rbp-48h]
  __int64 v21; // [rsp+60h] [rbp-40h]
  __int64 v22; // [rsp+68h] [rbp-38h]

  v5 = *a2;
  v15 = &v13;
  v12.m128i_i64[0] = 0;
  v13 = 0;
  v14 = 0;
  v16 = &v13;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  if ( sub_15603E0((_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 56), a3) )
  {
    v7 = sub_15603E0((_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56), a3);
    v8 = sub_155CEC0(a1, 9, v7);
    sub_1562E30(&v12, v8);
  }
  if ( sub_1560400((_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56), a3) )
  {
    v9 = sub_1560400((_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56), a3);
    v10 = sub_155CEC0(a1, 10, v9);
    sub_1562E30(&v12, v10);
  }
  v11 = *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (unsigned __int8)sub_1560260(&v11, a3, 20) )
    sub_15606E0(&v12, 20);
  if ( v12.m128i_i64[0] )
  {
    v11 = *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56);
    *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 56) = sub_1563330(&v11, a1, a3, &v12);
  }
  return sub_1A95860(v14);
}
