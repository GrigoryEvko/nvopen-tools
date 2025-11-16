// Function: sub_39405A0
// Address: 0x39405a0
//
__int64 __fastcall sub_39405A0(__int64 a1)
{
  __m128i *v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __m128i *v5; // rsi
  __int64 v6; // rcx
  const __m128i *v7; // rdx
  unsigned int v8; // r14d
  __int64 result; // rax
  __m128i *v10; // r14
  __int64 v11; // r15
  signed __int64 v12; // r12
  __int64 v13; // rax
  __m128i *v14; // rax
  unsigned int v15; // [rsp+0h] [rbp-70h] BYREF
  char v16; // [rsp+10h] [rbp-60h]
  __m128i v17; // [rsp+20h] [rbp-50h] BYREF
  char v18; // [rsp+30h] [rbp-40h]

  v2 = (__m128i *)&v15;
  sub_3940120((__int64)&v15, (_QWORD *)a1);
  if ( (v16 & 1) == 0 || (result = v15) == 0 )
  {
    v5 = *(__m128i **)(a1 + 88);
    v6 = v15;
    v7 = v5;
    if ( v15 > (unsigned __int64)((__int64)(*(_QWORD *)(a1 + 104) - (_QWORD)v5) >> 4) )
    {
      v2 = *(__m128i **)(a1 + 96);
      v10 = 0;
      v11 = v15;
      v12 = (char *)v2 - (char *)v5;
      if ( v15 )
      {
        v13 = sub_22077B0(16LL * v15);
        v5 = *(__m128i **)(a1 + 88);
        v2 = *(__m128i **)(a1 + 96);
        v10 = (__m128i *)v13;
        v7 = v5;
      }
      if ( v5 != v2 )
      {
        v14 = v10;
        v6 = (__int64)v10->m128i_i64 + (char *)v2 - (char *)v5;
        do
        {
          if ( v14 )
            *v14 = _mm_loadu_si128(v7);
          ++v14;
          ++v7;
        }
        while ( v14 != (__m128i *)v6 );
        v2 = v5;
      }
      if ( v2 )
      {
        v5 = (__m128i *)(*(_QWORD *)(a1 + 104) - (_QWORD)v2);
        j_j___libc_free_0((unsigned __int64)v2);
      }
      *(_QWORD *)(a1 + 88) = v10;
      *(_QWORD *)(a1 + 96) = (char *)v10 + v12;
      *(_QWORD *)(a1 + 104) = &v10[v11];
    }
    v8 = 0;
    if ( v15 )
    {
      while ( 1 )
      {
        v2 = &v17;
        sub_393EFD0(v17.m128i_i8, (_QWORD *)a1, (__int64)v7, v6, v3, v4);
        if ( (v18 & 1) != 0 )
        {
          result = v17.m128i_u32[0];
          if ( v17.m128i_i32[0] )
            break;
        }
        v5 = *(__m128i **)(a1 + 96);
        if ( v5 == *(__m128i **)(a1 + 104) )
        {
          v2 = (__m128i *)(a1 + 88);
          ++v8;
          sub_1516D00((const __m128i **)(a1 + 88), v5, &v17);
          if ( v15 <= v8 )
            goto LABEL_12;
        }
        else
        {
          if ( v5 )
          {
            *v5 = _mm_loadu_si128(&v17);
            v5 = *(__m128i **)(a1 + 96);
          }
          ++v5;
          ++v8;
          *(_QWORD *)(a1 + 96) = v5;
          if ( v15 <= v8 )
            goto LABEL_12;
        }
      }
    }
    else
    {
LABEL_12:
      sub_393D180((__int64)v2, (__int64)v5, (__int64)v7, v6, v3, v4);
      return 0;
    }
  }
  return result;
}
