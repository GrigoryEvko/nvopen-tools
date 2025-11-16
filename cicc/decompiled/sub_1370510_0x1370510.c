// Function: sub_1370510
// Address: 0x1370510
//
void __fastcall sub_1370510(char *src, char *a2)
{
  char *i; // rbx
  __int32 v3; // ecx
  unsigned __int32 v4; // r12d
  __m128i *v5; // rdx
  __int64 v6; // r15
  const __m128i *v7; // rax
  __m128i v8; // xmm0
  int v9; // [rsp-3Ch] [rbp-3Ch]

  if ( src != a2 )
  {
    for ( i = src + 16; i != a2; *((_QWORD *)src + 1) = v6 )
    {
      while ( 1 )
      {
        v4 = *((_DWORD *)i + 1);
        v3 = *(_DWORD *)i;
        v5 = (__m128i *)i;
        v6 = *((_QWORD *)i + 1);
        if ( v4 < *((_DWORD *)src + 1) )
          break;
        v7 = (const __m128i *)(i - 16);
        if ( v4 < *((_DWORD *)i - 3) )
        {
          do
          {
            v8 = _mm_loadu_si128(v7);
            v5 = (__m128i *)v7--;
            v7[2] = v8;
          }
          while ( v4 < v7->m128i_i32[1] );
        }
        i += 16;
        v5->m128i_i32[0] = v3;
        v5->m128i_i32[1] = v4;
        v5->m128i_i64[1] = v6;
        if ( i == a2 )
          return;
      }
      if ( src != i )
      {
        v9 = *(_DWORD *)i;
        memmove(src + 16, src, i - src);
        v3 = v9;
      }
      i += 16;
      *(_DWORD *)src = v3;
      *((_DWORD *)src + 1) = v4;
    }
  }
}
