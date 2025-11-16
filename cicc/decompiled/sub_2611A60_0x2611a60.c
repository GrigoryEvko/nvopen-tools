// Function: sub_2611A60
// Address: 0x2611a60
//
_QWORD *__fastcall sub_2611A60(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rdi
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  _QWORD *result; // rax
  __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  __int16 v12; // [rsp+20h] [rbp-40h]

  v2 = a1 + 72;
  sub_B2CA40(a1, 1);
  v3 = *(_QWORD *)(a1 + 80);
  if ( a1 + 72 != v3 )
  {
    do
    {
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      sub_AA5450((_QWORD *)(v4 - 24));
    }
    while ( v2 != v3 );
  }
  v12 = 257;
  v5 = sub_B2BE50(a1);
  v6 = sub_22077B0(0x50u);
  v7 = v6;
  if ( v6 )
    sub_AA4D50(v6, v5, (__int64)&v10, a1, 0);
  v8 = sub_B2BE50(a1);
  sub_B43C20((__int64)&v10, v7);
  result = sub_BD2C40(72, unk_3F148B8);
  if ( result )
    return sub_B4C8A0((__int64)result, v8, v10, v11);
  return result;
}
