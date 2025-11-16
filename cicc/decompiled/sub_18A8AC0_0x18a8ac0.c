// Function: sub_18A8AC0
// Address: 0x18a8ac0
//
unsigned __int64 __fastcall sub_18A8AC0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rsi
  __m128i *v8; // r8
  __m128i *v9; // rcx
  __m128i *v10; // rax
  unsigned __int8 v12; // dl
  int v13; // eax

  if ( *(_QWORD *)(a1 + 568) )
  {
    sub_18A8930((_QWORD *)(a1 + 528), a2);
    return ((unsigned __int64)v12 << 32) | 1;
  }
  else
  {
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(__m128i **)a1;
    v9 = (__m128i *)(*(_QWORD *)a1 + 16 * v7);
    if ( *(__m128i **)a1 == v9 )
      goto LABEL_10;
    v10 = *(__m128i **)a1;
    while ( 1 )
    {
      if ( v10->m128i_i64[0] == a2->m128i_i64[0] )
      {
        a6 = a2->m128i_i64[1];
        if ( v10->m128i_i64[1] == a6 )
          break;
      }
      if ( v9 == ++v10 )
        goto LABEL_10;
    }
    if ( v10 == v9 )
    {
LABEL_10:
      if ( v7 > 0x1F )
      {
        while ( 1 )
        {
          sub_18A8930((_QWORD *)(a1 + 528), &v8[v7 - 1]);
          v13 = *(_DWORD *)(a1 + 8);
          *(_DWORD *)(a1 + 8) = v13 - 1;
          if ( v13 == 1 )
            break;
          v8 = *(__m128i **)a1;
          v7 = (unsigned int)(v13 - 1);
        }
        sub_18A8930((_QWORD *)(a1 + 528), a2);
        return 0x100000001LL;
      }
      else
      {
        if ( *(_DWORD *)(a1 + 8) >= *(_DWORD *)(a1 + 12) )
        {
          sub_16CD150(a1, (const void *)(a1 + 16), 0, 16, (int)v8, a6);
          v9 = (__m128i *)(*(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8));
        }
        *v9 = _mm_loadu_si128(a2);
        ++*(_DWORD *)(a1 + 8);
        return 0x100000001LL;
      }
    }
    else
    {
      return 1;
    }
  }
}
