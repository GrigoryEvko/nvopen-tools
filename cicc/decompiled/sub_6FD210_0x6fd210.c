// Function: sub_6FD210
// Address: 0x6fd210
//
__int64 __fastcall sub_6FD210(__m128i *a1, __int64 a2)
{
  __int32 v2; // r15d
  __int16 v3; // r14
  __int32 v4; // r13d
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __int64 i; // rax
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // [rsp+0h] [rbp-40h]
  unsigned __int16 v15; // [rsp+Eh] [rbp-32h]

  v2 = a1[4].m128i_i32[1];
  v3 = a1[4].m128i_i16[4];
  v4 = a1[4].m128i_i32[3];
  v15 = a1[5].m128i_u16[0];
  result = sub_8D3D40(a1->m128i_i64[0]);
  if ( !(_DWORD)result )
  {
    for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v10 = *(_QWORD *)(i + 160);
    if ( a1->m128i_i64[0] != v10 )
    {
      v14 = *(_QWORD *)(i + 160);
      v11 = sub_8D97D0(a1->m128i_i64[0], v14, 32, v7, v10);
      v10 = v14;
      if ( !v11 )
        sub_6FC3F0(v14, a1, 1u);
    }
    v12 = sub_6F6F40(a1, 0, v6, v7, v10, v8);
    v13 = sub_73DBF0(31, a2, v12);
    sub_6E70E0((__int64 *)v13, (__int64)a1);
    *(_BYTE *)(v13 + 27) |= 2u;
    a1[4].m128i_i32[1] = v2;
    a1[4].m128i_i16[4] = v3;
    a1[4].m128i_i32[3] = v4;
    a1[5].m128i_i16[0] = v15;
    return v15;
  }
  return result;
}
