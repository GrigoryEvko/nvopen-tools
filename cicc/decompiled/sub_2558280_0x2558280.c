// Function: sub_2558280
// Address: 0x2558280
//
__m128i *__fastcall sub_2558280(__m128i *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rcx
  int v5; // eax
  __m128i *v6; // rax
  unsigned __int64 v7; // rcx
  _BYTE *v9; // [rsp+0h] [rbp-60h] BYREF
  int v10; // [rsp+8h] [rbp-58h]
  _QWORD v11[2]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v12[2]; // [rsp+20h] [rbp-40h] BYREF
  _OWORD v13[3]; // [rsp+30h] [rbp-30h] BYREF

  v2 = *(unsigned int *)(a2 + 112);
  if ( v2 <= 9 )
  {
    a2 = 1;
  }
  else if ( v2 <= 0x63 )
  {
    a2 = 2;
  }
  else if ( v2 <= 0x3E7 )
  {
    a2 = 3;
  }
  else if ( v2 <= 0x270F )
  {
    a2 = 4;
  }
  else
  {
    v3 = *(unsigned int *)(a2 + 112);
    LODWORD(a2) = 1;
    while ( 1 )
    {
      v4 = v3;
      v5 = a2;
      a2 = (unsigned int)(a2 + 4);
      v3 /= 0x2710u;
      if ( v4 <= 0x1869F )
        break;
      if ( v4 <= 0xF423F )
      {
        a2 = (unsigned int)(v5 + 5);
        break;
      }
      if ( v4 <= (unsigned __int64)&loc_98967F )
      {
        a2 = (unsigned int)(v5 + 6);
        break;
      }
      if ( v4 <= 0x5F5E0FF )
      {
        a2 = (unsigned int)(v5 + 7);
        break;
      }
    }
  }
  v9 = v11;
  sub_2240A50((__int64 *)&v9, a2, 0);
  sub_1249540(v9, v10, v2);
  v6 = (__m128i *)sub_2241130((unsigned __int64 *)&v9, 0, 0, "#queries(", 9u);
  v12[0] = (unsigned __int64)v13;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v13[0] = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v12[0] = v6->m128i_i64[0];
    *(_QWORD *)&v13[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_u64[1];
  v6[1].m128i_i8[0] = 0;
  v12[1] = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  sub_94F930(a1, (__int64)v12, ")");
  if ( (_OWORD *)v12[0] != v13 )
    j_j___libc_free_0(v12[0]);
  if ( v9 != (_BYTE *)v11 )
    j_j___libc_free_0((unsigned __int64)v9);
  return a1;
}
