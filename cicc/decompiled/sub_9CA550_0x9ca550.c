// Function: sub_9CA550
// Address: 0x9ca550
//
unsigned __int64 __fastcall sub_9CA550(__int64 a1, unsigned __int64 a2)
{
  bool v2; // zf
  _QWORD *v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v8; // [rsp+8h] [rbp-58h] BYREF
  __m128i v9; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v10; // [rsp+20h] [rbp-40h]
  _QWORD *v11; // [rsp+28h] [rbp-38h]
  __int64 v12; // [rsp+30h] [rbp-30h]

  v2 = *(_BYTE *)(a1 + 343) == 0;
  v8 = a2;
  if ( v2 )
  {
    v9.m128i_i64[1] = 0;
    v9.m128i_i64[0] = (__int64)byte_3F871B3;
  }
  else
  {
    v9.m128i_i64[0] = 0;
  }
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v3 = sub_9CA390((_QWORD *)a1, &v8, &v9);
  v4 = v11;
  v5 = v10;
  v6 = (unsigned __int64)(v3 + 4);
  if ( v11 != v10 )
  {
    do
    {
      if ( *v5 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v5 + 8LL))(*v5);
      ++v5;
    }
    while ( v4 != v5 );
    v5 = v10;
  }
  if ( v5 )
    j_j___libc_free_0(v5, v12 - (_QWORD)v5);
  return *(unsigned __int8 *)(a1 + 343) | v6 & 0xFFFFFFFFFFFFFFF8LL;
}
