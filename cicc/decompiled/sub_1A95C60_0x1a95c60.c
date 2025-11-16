// Function: sub_1A95C60
// Address: 0x1a95c60
//
__int64 __fastcall sub_1A95C60(__int64 *a1, __int64 a2, int a3)
{
  _QWORD *v3; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // [rsp+8h] [rbp-98h] BYREF
  __m128i v11; // [rsp+10h] [rbp-90h] BYREF
  int v12; // [rsp+20h] [rbp-80h] BYREF
  _QWORD *v13; // [rsp+28h] [rbp-78h]
  int *v14; // [rsp+30h] [rbp-70h]
  int *v15; // [rsp+38h] [rbp-68h]
  __int64 v16; // [rsp+40h] [rbp-60h]
  __int64 v17; // [rsp+48h] [rbp-58h]
  __int64 v18; // [rsp+50h] [rbp-50h]
  __int64 v19; // [rsp+58h] [rbp-48h]
  __int64 v20; // [rsp+60h] [rbp-40h]
  __int64 v21; // [rsp+68h] [rbp-38h]

  v3 = (_QWORD *)(a2 + 112);
  v12 = 0;
  v11.m128i_i64[0] = 0;
  v13 = 0;
  v14 = &v12;
  v15 = &v12;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  if ( sub_15603E0((_QWORD *)(a2 + 112), a3) )
  {
    v6 = sub_15603E0(v3, a3);
    v7 = sub_155CEC0(a1, 9, v6);
    sub_1562E30(&v11, v7);
    if ( !sub_1560400(v3, a3) )
      goto LABEL_3;
  }
  else if ( !sub_1560400(v3, a3) )
  {
    goto LABEL_3;
  }
  v8 = sub_1560400(v3, a3);
  v9 = sub_155CEC0(a1, 10, v8);
  sub_1562E30(&v11, v9);
LABEL_3:
  v10 = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_1560260(&v10, a3, 20) )
    sub_15606E0(&v11, 20);
  if ( v11.m128i_i64[0] )
  {
    v10 = *(_QWORD *)(a2 + 112);
    *(_QWORD *)(a2 + 112) = sub_1563330(&v10, a1, a3, &v11);
  }
  return sub_1A95860(v13);
}
