// Function: sub_2554AF0
// Address: 0x2554af0
//
__m128i *__fastcall sub_2554AF0(__m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 *(__fastcall *v3)(__int64 *); // rax
  _BYTE *v5[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h] BYREF
  __int64 v7[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v8; // [rsp+30h] [rbp-30h] BYREF

  v2 = sub_2509800((_QWORD *)(*(_QWORD *)a2 + 72LL));
  sub_2509010(v7, v2);
  v3 = *(__int64 *(__fastcall **)(__int64 *))(**(_QWORD **)a2 + 72LL);
  if ( v3 == sub_253C1A0 )
    sub_253C590((__int64 *)v5, "AAValueConstantRange");
  else
    v3((__int64 *)v5);
  sub_253A410(a1, v5[0], (size_t)v5[1], (unsigned __int64 *)v7);
  if ( (__int64 *)v5[0] != &v6 )
    j_j___libc_free_0((unsigned __int64)v5[0]);
  if ( (__int64 *)v7[0] != &v8 )
    j_j___libc_free_0(v7[0]);
  return a1;
}
