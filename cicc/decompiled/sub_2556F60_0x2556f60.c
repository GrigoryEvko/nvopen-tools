// Function: sub_2556F60
// Address: 0x2556f60
//
__m128i *__fastcall sub_2556F60(__m128i *a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rcx
  int v6; // r8d
  __m128i *v7; // rax
  unsigned __int64 v8; // rcx
  _BYTE *v10; // [rsp+0h] [rbp-60h] BYREF
  int v11; // [rsp+8h] [rbp-58h]
  _QWORD v12[2]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v13[2]; // [rsp+20h] [rbp-40h] BYREF
  _OWORD v14[3]; // [rsp+30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a2 + 97) )
  {
    v2 = *(_DWORD *)(a2 + 100);
    if ( v2 == -1 )
    {
      sub_253C590((__int64 *)&v10, "none");
    }
    else
    {
      if ( v2 <= 9 )
      {
        v4 = 1;
      }
      else if ( v2 <= 0x63 )
      {
        v4 = 2;
      }
      else if ( v2 <= 0x3E7 )
      {
        v4 = 3;
      }
      else
      {
        v3 = v2;
        if ( v2 <= 0x270F )
        {
          v4 = 4;
        }
        else
        {
          LODWORD(v4) = 1;
          while ( 1 )
          {
            v5 = v3;
            v6 = v4;
            v4 = (unsigned int)(v4 + 4);
            v3 /= 0x2710u;
            if ( v5 <= 0x1869F )
              break;
            if ( (unsigned int)v3 <= 0x63 )
            {
              v4 = (unsigned int)(v6 + 5);
              break;
            }
            if ( (unsigned int)v3 <= 0x3E7 )
            {
              v4 = (unsigned int)(v6 + 6);
              break;
            }
            if ( (unsigned int)v3 <= 0x270F )
            {
              v4 = (unsigned int)(v6 + 7);
              break;
            }
          }
        }
      }
      v10 = v12;
      sub_2240A50((__int64 *)&v10, v4, 0);
      sub_2554A60(v10, v11, v2);
    }
    v7 = (__m128i *)sub_2241130((unsigned __int64 *)&v10, 0, 0, "addrspace(", 0xAu);
    v13[0] = (unsigned __int64)v14;
    if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
    {
      v14[0] = _mm_loadu_si128(v7 + 1);
    }
    else
    {
      v13[0] = v7->m128i_i64[0];
      *(_QWORD *)&v14[0] = v7[1].m128i_i64[0];
    }
    v8 = v7->m128i_u64[1];
    v7[1].m128i_i8[0] = 0;
    v13[1] = v8;
    v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
    v7->m128i_i64[1] = 0;
    sub_94F930(a1, (__int64)v13, ")");
    if ( (_OWORD *)v13[0] != v14 )
      j_j___libc_free_0(v13[0]);
    if ( v10 != (_BYTE *)v12 )
      j_j___libc_free_0((unsigned __int64)v10);
    return a1;
  }
  else
  {
    sub_253C590(a1->m128i_i64, "addrspace(<invalid>)");
    return a1;
  }
}
