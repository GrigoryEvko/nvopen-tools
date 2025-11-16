// Function: sub_1801990
// Address: 0x1801990
//
_QWORD *__fastcall sub_1801990(__int64 *a1, char *a2, signed __int64 a3, char a4)
{
  __int64 *v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  _QWORD *v8; // r12
  const char *v10; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+10h] [rbp-40h]
  char v12; // [rsp+11h] [rbp-3Fh]

  v5 = (__int64 *)sub_15996B0(*a1, a2, a3, 1);
  v6 = *v5;
  v7 = (__int64)v5;
  v12 = 1;
  v10 = "___asan_gen_";
  v11 = 3;
  v8 = sub_1648A60(88, 1u);
  if ( v8 )
    sub_15E51E0((__int64)v8, (__int64)a1, v6, 1, 8, v7, (__int64)&v10, 0, 0, 0, 0);
  if ( a4 )
    *((_BYTE *)v8 + 32) = v8[4] & 0x3F | 0x80;
  sub_15E4CC0((__int64)v8, 1u);
  return v8;
}
