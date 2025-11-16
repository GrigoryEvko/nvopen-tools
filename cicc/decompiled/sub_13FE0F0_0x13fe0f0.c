// Function: sub_13FE0F0
// Address: 0x13fe0f0
//
__int64 __fastcall sub_13FE0F0(__int64 a1)
{
  __m128i *v2; // r14
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 *v8; // rax
  char v9; // dl
  __int64 v10; // rax
  __int64 *v11; // rcx
  unsigned int v12; // edi
  __int64 *v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rdx
  __m128i v16; // [rsp+0h] [rbp-40h] BYREF
  __int64 v17; // [rsp+10h] [rbp-30h]

  v2 = *(__m128i **)(a1 + 112);
  while ( 1 )
  {
    v3 = sub_157EBA0(v2[-2].m128i_i64[1]);
    result = 0;
    if ( v3 )
    {
      result = sub_15F4D60(v3);
      v2 = *(__m128i **)(a1 + 112);
    }
    v5 = v2[-1].m128i_u32[2];
    if ( (_DWORD)v5 == (_DWORD)result )
      return result;
    v6 = v2[-1].m128i_i64[0];
    v2[-1].m128i_i32[2] = v5 + 1;
    v7 = sub_15F4DF0(v6, v5);
    v8 = *(__int64 **)(a1 + 8);
    if ( *(__int64 **)(a1 + 16) != v8 )
      goto LABEL_6;
    v11 = &v8[*(unsigned int *)(a1 + 28)];
    v12 = *(_DWORD *)(a1 + 28);
    if ( v8 == v11 )
    {
LABEL_21:
      if ( v12 < *(_DWORD *)(a1 + 24) )
      {
        *(_DWORD *)(a1 + 28) = v12 + 1;
        *v11 = v7;
        v2 = *(__m128i **)(a1 + 112);
        ++*(_QWORD *)a1;
        goto LABEL_7;
      }
LABEL_6:
      sub_16CCBA0(a1, v7);
      v2 = *(__m128i **)(a1 + 112);
      if ( v9 )
      {
LABEL_7:
        v10 = sub_157EBA0(v7);
        v16.m128i_i64[0] = v7;
        v16.m128i_i64[1] = v10;
        LODWORD(v17) = 0;
        if ( v2 == *(__m128i **)(a1 + 120) )
        {
          sub_13FDF40((const __m128i **)(a1 + 104), v2, &v16);
          v2 = *(__m128i **)(a1 + 112);
        }
        else
        {
          if ( v2 )
          {
            *v2 = _mm_loadu_si128(&v16);
            v2[1].m128i_i64[0] = v17;
            v2 = *(__m128i **)(a1 + 112);
          }
          v2 = (__m128i *)((char *)v2 + 24);
          *(_QWORD *)(a1 + 112) = v2;
        }
      }
    }
    else
    {
      v13 = 0;
      while ( 2 )
      {
        v14 = *v8;
        if ( v7 != *v8 )
        {
          while ( v14 == -2 )
          {
            v15 = v8 + 1;
            v13 = v8;
            if ( v8 + 1 == v11 )
              goto LABEL_17;
            ++v8;
            v14 = *v15;
            if ( v7 == v14 )
              goto LABEL_20;
          }
          if ( v11 != ++v8 )
            continue;
          if ( v13 )
          {
LABEL_17:
            *v13 = v7;
            v2 = *(__m128i **)(a1 + 112);
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_7;
          }
          goto LABEL_21;
        }
        break;
      }
LABEL_20:
      v2 = *(__m128i **)(a1 + 112);
    }
  }
}
