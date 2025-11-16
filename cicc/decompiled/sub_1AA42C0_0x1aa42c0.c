// Function: sub_1AA42C0
// Address: 0x1aa42c0
//
void __fastcall sub_1AA42C0(char *src, char *a2, unsigned __int8 (__fastcall *a3)(__m128i *, __int8 *))
{
  const __m128i *v4; // rbx
  unsigned __int8 i; // al
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r15
  __int32 v13; // r14d
  const __m128i *v14; // rdi
  __int64 v15; // [rsp-68h] [rbp-68h]
  __int64 v16; // [rsp-60h] [rbp-60h]
  __int64 v17; // [rsp-58h] [rbp-58h]
  __int64 v18; // [rsp-50h] [rbp-50h]
  __int64 v19; // [rsp-48h] [rbp-48h]

  if ( src != a2 )
  {
    v4 = (const __m128i *)(src + 56);
    if ( a2 != src + 56 )
    {
      for ( i = a3((__m128i *)v4, src); ; i = a3((__m128i *)v4, src) )
      {
        if ( i )
        {
          v7 = v4->m128i_i64[0];
          v8 = v4->m128i_i64[1];
          v9 = v4[1].m128i_i64[0];
          v10 = v4[1].m128i_i64[1];
          v11 = v4[2].m128i_i64[0];
          v12 = v4[2].m128i_i64[1];
          v13 = v4[3].m128i_i32[0];
          if ( src != (char *)v4 )
          {
            v15 = v4->m128i_i64[1];
            v16 = v4[2].m128i_i64[0];
            v17 = v4[1].m128i_i64[1];
            v18 = v4->m128i_i64[0];
            v19 = v4[1].m128i_i64[0];
            memmove(src + 56, src, (char *)v4 - src);
            v8 = v15;
            v11 = v16;
            v10 = v17;
            v7 = v18;
            v9 = v19;
          }
          *(_QWORD *)src = v7;
          v4 = (const __m128i *)((char *)v4 + 56);
          *((_QWORD *)src + 1) = v8;
          *((_QWORD *)src + 2) = v9;
          *((_QWORD *)src + 3) = v10;
          *((_QWORD *)src + 4) = v11;
          *((_QWORD *)src + 5) = v12;
          *((_DWORD *)src + 12) = v13;
          if ( a2 == (char *)v4 )
            return;
        }
        else
        {
          v14 = v4;
          v4 = (const __m128i *)((char *)v4 + 56);
          sub_1AA4200(v14, a3);
          if ( a2 == (char *)v4 )
            return;
        }
      }
    }
  }
}
