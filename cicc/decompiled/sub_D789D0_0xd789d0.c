// Function: sub_D789D0
// Address: 0xd789d0
//
unsigned __int64 __fastcall sub_D789D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  bool v3; // zf
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // r14
  unsigned __int64 v9; // [rsp+8h] [rbp-78h]
  __int64 v10; // [rsp+18h] [rbp-68h] BYREF
  __m128i v11; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v12; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+40h] [rbp-40h]

  sub_B2F930(&v11, a2);
  v2 = sub_B2F650(v11.m128i_i64[0], v11.m128i_i64[1]);
  if ( (_QWORD **)v11.m128i_i64[0] != &v12 )
    j_j___libc_free_0(v11.m128i_i64[0], (char *)v12 + 1);
  v3 = *(_BYTE *)(a1 + 343) == 0;
  v10 = v2;
  if ( v3 )
  {
    v11.m128i_i64[1] = 0;
    v11.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v11.m128i_i64[0] = 0;
  }
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v4 = sub_9CA390((_QWORD *)a1, (unsigned __int64 *)&v10, &v11);
  v5 = v13;
  v6 = v12;
  v7 = v4;
  v9 = (unsigned __int64)(v4 + 4);
  if ( v13 != v12 )
  {
    do
    {
      if ( *v6 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v6 + 8LL))(*v6);
      ++v6;
    }
    while ( v5 != v6 );
    v6 = v12;
  }
  if ( v6 )
    j_j___libc_free_0(v6, v14 - (_QWORD)v6);
  v7[5] = a2;
  return v9 & 0xFFFFFFFFFFFFFFF8LL | *(unsigned __int8 *)(a1 + 343);
}
