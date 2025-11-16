// Function: sub_3099D80
// Address: 0x3099d80
//
__int64 __fastcall sub_3099D80(__int64 *a1, _QWORD **a2, __int64 a3, char a4)
{
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD **v7; // r14
  unsigned int v8; // r12d
  char v10; // [rsp+Ch] [rbp-54h] BYREF
  _QWORD **v11; // [rsp+18h] [rbp-48h] BYREF
  __m128i v12; // [rsp+20h] [rbp-40h] BYREF
  __int64 (__fastcall *v13)(__m128i *, __m128i *, int); // [rsp+30h] [rbp-30h]
  _BYTE *(__fastcall *v14)(_QWORD *, __int64, __int64); // [rsp+38h] [rbp-28h]

  v12.m128i_i64[0] = (__int64)&v10;
  v10 = a4;
  v14 = sub_3099C20;
  v11 = a2;
  v13 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_3099A00;
  v4 = sub_E49E40(a1, (__int64)&v11, 3u, &v12);
  v7 = v11;
  v8 = v4;
  if ( v11 )
  {
    sub_BA9C10(v11, (__int64)&v11, v5, v6);
    j_j___libc_free_0((unsigned __int64)v7);
  }
  if ( v13 )
    v13(&v12, &v12, 3);
  return v8;
}
