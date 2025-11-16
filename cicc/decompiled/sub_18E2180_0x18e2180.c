// Function: sub_18E2180
// Address: 0x18e2180
//
void __fastcall sub_18E2180(__int64 a1)
{
  __m128i *v2; // r12
  __int64 i; // r14
  __int64 v4; // rbx
  _QWORD *v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 *v8; // rax
  char v9; // dl
  __int64 v10; // r14
  __int64 *v11; // rcx
  unsigned int v12; // r8d
  __int64 *v13; // rsi
  __m128i v14; // [rsp+0h] [rbp-30h] BYREF

LABEL_1:
  v2 = *(__m128i **)(a1 + 16);
LABEL_2:
  for ( i = v2[-1].m128i_i64[1]; i; i = v2[-1].m128i_i64[1] )
  {
    v4 = *(_QWORD *)(i + 8);
    for ( v2[-1].m128i_i64[1] = v4; v4; v2[-1].m128i_i64[1] = v4 )
    {
      if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v4) + 16) - 25) <= 9u )
        break;
      v4 = *(_QWORD *)(v4 + 8);
    }
    v5 = sub_1648700(i);
    v6 = *(_QWORD *)a1;
    v7 = v5[5];
    v8 = *(__int64 **)(*(_QWORD *)a1 + 8LL);
    if ( *(__int64 **)(*(_QWORD *)a1 + 16LL) != v8 )
      goto LABEL_6;
    v11 = &v8[*(unsigned int *)(v6 + 28)];
    v12 = *(_DWORD *)(v6 + 28);
    if ( v8 != v11 )
    {
      v13 = 0;
      while ( v7 != *v8 )
      {
        if ( *v8 == -2 )
        {
          v13 = v8;
          if ( v8 + 1 == v11 )
            goto LABEL_22;
          ++v8;
        }
        else if ( v11 == ++v8 )
        {
          if ( !v13 )
            goto LABEL_28;
LABEL_22:
          *v13 = v7;
          --*(_DWORD *)(v6 + 32);
          ++*(_QWORD *)v6;
          v2 = *(__m128i **)(a1 + 16);
          goto LABEL_7;
        }
      }
      goto LABEL_1;
    }
LABEL_28:
    if ( v12 < *(_DWORD *)(v6 + 24) )
    {
      *(_DWORD *)(v6 + 28) = v12 + 1;
      *v11 = v7;
      ++*(_QWORD *)v6;
      v2 = *(__m128i **)(a1 + 16);
    }
    else
    {
LABEL_6:
      sub_16CCBA0(v6, v7);
      v2 = *(__m128i **)(a1 + 16);
      if ( !v9 )
        goto LABEL_2;
    }
LABEL_7:
    v10 = *(_QWORD *)(v7 + 8);
    if ( v10 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) > 9u )
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
        {
          v14 = (__m128i)(unsigned __int64)v7;
          if ( v2 != *(__m128i **)(a1 + 24) )
            goto LABEL_10;
          goto LABEL_27;
        }
      }
    }
    v14.m128i_i64[0] = v7;
    v14.m128i_i64[1] = v10;
    if ( v2 == *(__m128i **)(a1 + 24) )
    {
LABEL_27:
      sub_18E2000((const __m128i **)(a1 + 8), v2, &v14);
      v2 = *(__m128i **)(a1 + 16);
      goto LABEL_2;
    }
LABEL_10:
    if ( v2 )
    {
      *v2 = _mm_loadu_si128(&v14);
      v2 = *(__m128i **)(a1 + 16);
    }
    *(_QWORD *)(a1 + 16) = ++v2;
  }
}
