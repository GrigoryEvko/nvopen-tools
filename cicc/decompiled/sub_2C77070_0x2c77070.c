// Function: sub_2C77070
// Address: 0x2c77070
//
struct __jmp_buf_tag *__fastcall sub_2C77070(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 v6; // r12
  __int64 i; // r13
  struct __jmp_buf_tag *result; // rax
  const char *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __m128i si128; // xmm0
  __int64 v13; // rcx
  _BYTE *v14; // rax
  __int64 v15; // rdi

  if ( *(_BYTE *)a2 == 5 )
  {
    v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( v5 )
    {
      v6 = v5 - 1;
      for ( i = 0; ; ++i )
      {
        sub_2C77070(a1, *(_QWORD *)(a2 + 32 * (i - v5)), a3);
        if ( i == v6 )
          break;
        v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      }
    }
  }
  result = *(struct __jmp_buf_tag **)(a2 + 8);
  if ( LOBYTE(result->__jmpbuf[1]) == 14 )
  {
    result = (struct __jmp_buf_tag *)LODWORD(result->__jmpbuf[1]);
    if ( (unsigned int)result > 0x1FF && (unsigned int)result >> 8 != 4 )
    {
      v9 = (const char *)a3;
      v10 = sub_2C767C0(a1, a3, 0);
      v11 = *(_QWORD *)(v10 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v10 + 24) - v11) <= 0x35 )
      {
        v9 = "Invalid address space for global constant initializer\n";
        sub_CB6200(v10, "Invalid address space for global constant initializer\n", 0x36u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42D0500);
        v13 = 2674;
        *(_DWORD *)(v11 + 48) = 1702521196;
        *(_WORD *)(v11 + 52) = 2674;
        *(__m128i *)v11 = si128;
        *(__m128i *)(v11 + 16) = _mm_load_si128((const __m128i *)&xmmword_42D0510);
        *(__m128i *)(v11 + 32) = _mm_load_si128((const __m128i *)&xmmword_42D0520);
        *(_QWORD *)(v10 + 32) += 54LL;
      }
      v14 = *(_BYTE **)(a1 + 16);
      if ( v14 )
        *v14 = 0;
      result = (struct __jmp_buf_tag *)*(unsigned int *)(a1 + 4);
      if ( !(_DWORD)result )
      {
        v15 = *(_QWORD *)(a1 + 24);
        if ( *(_QWORD *)(v15 + 32) != *(_QWORD *)(v15 + 16) )
        {
          sub_CB5AE0((__int64 *)v15);
          v15 = *(_QWORD *)(a1 + 24);
        }
        return sub_CEB520(*(_QWORD **)(v15 + 48), (__int64)v9, v11, (char *)v13);
      }
    }
  }
  return result;
}
