// Function: sub_1F0BDE0
// Address: 0x1f0bde0
//
_QWORD *__fastcall sub_1F0BDE0(__int64 a1)
{
  void (__fastcall *v1)(__int64, _QWORD **, _QWORD **); // r13
  __m128i *v2; // rax
  __int64 v3; // rcx
  _QWORD *result; // rax
  _QWORD *v5; // [rsp+0h] [rbp-C0h] BYREF
  __int16 v6; // [rsp+10h] [rbp-B0h]
  _QWORD *v7; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v8; // [rsp+30h] [rbp-90h]
  _QWORD v9[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v10; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v11[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v12[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v13[2]; // [rsp+80h] [rbp-40h] BYREF
  _OWORD v14[3]; // [rsp+90h] [rbp-30h] BYREF

  v1 = *(void (__fastcall **)(__int64, _QWORD **, _QWORD **))(*(_QWORD *)a1 + 16LL);
  (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)a1 + 48LL))(v11, a1);
  v2 = (__m128i *)sub_2241130(v11, 0, 0, "Scheduling-Units Graph for ", 27);
  v13[0] = v14;
  if ( (__m128i *)v2->m128i_i64[0] == &v2[1] )
  {
    v14[0] = _mm_loadu_si128(v2 + 1);
  }
  else
  {
    v13[0] = v2->m128i_i64[0];
    *(_QWORD *)&v14[0] = v2[1].m128i_i64[0];
  }
  v3 = v2->m128i_i64[1];
  v2[1].m128i_i8[0] = 0;
  v13[1] = v3;
  v2->m128i_i64[0] = (__int64)v2[1].m128i_i64;
  v2->m128i_i64[1] = 0;
  v8 = 260;
  v7 = v13;
  (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)a1 + 48LL))(v9, a1);
  v5 = v9;
  v6 = 260;
  v1(a1, &v5, &v7);
  if ( (__int64 *)v9[0] != &v10 )
    j_j___libc_free_0(v9[0], v10 + 1);
  if ( (_OWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], *(_QWORD *)&v14[0] + 1LL);
  result = v12;
  if ( (_QWORD *)v11[0] != v12 )
    return (_QWORD *)j_j___libc_free_0(v11[0], v12[0] + 1LL);
  return result;
}
