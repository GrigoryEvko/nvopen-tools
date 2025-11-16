// Function: sub_7FA9D0
// Address: 0x7fa9d0
//
_BYTE *__fastcall sub_7FA9D0(_QWORD *a1, _QWORD *a2, int *a3)
{
  _QWORD *v4; // rax
  _BYTE *v5; // rbx
  __int64 v6; // rax
  __m128i *v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax

  v4 = sub_72BA30(byte_4F06A51[0]);
  v5 = sub_73E130(a2, (__int64)v4);
  v6 = sub_7E1C10();
  v7 = (__m128i *)sub_73E130(a1, v6);
  v8 = sub_73A830(0, 5u);
  v7[1].m128i_i64[0] = (__int64)v8;
  v8[2] = v5;
  v9 = sub_7E1C10();
  v10 = sub_7F89D0("__gen_nvvm_memset", &qword_4F18AA8, v9, v7);
  v11 = qword_4F18AA8;
  *(_BYTE *)(qword_4F18AA8 + 198) |= 0x18u;
  *(_BYTE *)(v11 - 8) |= 0x10u;
  return sub_7E69E0(v10, a3);
}
