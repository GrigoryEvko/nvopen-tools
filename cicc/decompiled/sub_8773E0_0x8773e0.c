// Function: sub_8773E0
// Address: 0x8773e0
//
_QWORD *__fastcall sub_8773E0(const __m128i *a1, char *a2, __int64 a3, int a4, int a5)
{
  char *v6; // r13
  __int64 v8; // r14
  __int64 *v9; // rdi
  __int64 v10; // rax
  char i; // dl
  _BYTE v15[160]; // [rsp+20h] [rbp-230h] BYREF
  _QWORD v16[50]; // [rsp+C0h] [rbp-190h] BYREF

  v6 = (char *)(a3 + 64);
  v8 = qword_4D03C50;
  sub_6E1E00(4u, (__int64)v15, 0, 0);
  sub_6F8E70(a3, v6, v6, v16, 0);
  v9 = v16;
  sub_7029D0(v9, a2, 0, 0, (__int64)v9, (__int64)a1);
  if ( a4 && a1[1].m128i_i8[0] )
  {
    v10 = a1->m128i_i64[0];
    for ( i = *(_BYTE *)(a1->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    if ( i )
    {
      a2 = v6;
      v9 = (__int64 *)a1;
      sub_6980A0(a1, v6, 0, 0, 1, a5);
    }
  }
  sub_6E2B30((__int64)v9, (__int64)a2);
  qword_4D03C50 = v8;
  return &qword_4D03C50;
}
