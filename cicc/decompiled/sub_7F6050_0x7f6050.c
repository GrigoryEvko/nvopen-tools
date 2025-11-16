// Function: sub_7F6050
// Address: 0x7f6050
//
void *__fastcall sub_7F6050(const __m128i *a1)
{
  _QWORD *v1; // r14
  __int64 v2; // rax
  __int64 v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // rdi
  _QWORD *v6; // rbx
  __int64 v7; // rsi
  _BYTE *v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  void *v11; // r13
  __int64 v12; // rax
  _BYTE *v13; // rax

  v1 = sub_7E8090(a1, 1u);
  v2 = sub_8D46C0(a1->m128i_i64[0]);
  v3 = sub_691620(v2);
  v4 = sub_72BA30(byte_4F06A51[0]);
  if ( !sub_7F5F50(v3, 0) )
    sub_721090();
  v5 = v4[16];
  if ( dword_4F06880 )
    v5 *= 2;
  v6 = sub_73A830(v5, byte_4F06A51[0]);
  v7 = sub_7E1C30();
  v8 = sub_73E110((__int64)v1, v7);
  *((_QWORD *)v8 + 2) = v6;
  v9 = (__int64)v8;
  v10 = sub_7E1C30();
  v11 = sub_73DBF0(0x33u, v10, v9);
  v12 = sub_72D2E0(v4);
  v13 = sub_73E110((__int64)v11, v12);
  return sub_73DBF0(3u, (__int64)v4, (__int64)v13);
}
