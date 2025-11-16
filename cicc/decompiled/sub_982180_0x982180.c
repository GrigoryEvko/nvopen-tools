// Function: sub_982180
// Address: 0x982180
//
void __fastcall sub_982180(char *src, char *a2, __int64 (__fastcall *a3)(__m128i *, const __m128i *))
{
  const __m128i *v4; // rbx
  char i; // al
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // r8
  __int32 v11; // ecx
  char v12; // r15
  __int64 v13; // r13
  const __m128i *v14; // rdi
  __int64 v15; // [rsp-70h] [rbp-70h]
  __int64 v16; // [rsp-68h] [rbp-68h]
  __int64 v17; // [rsp-60h] [rbp-60h]
  __int64 v18; // [rsp-58h] [rbp-58h]
  __int64 v19; // [rsp-50h] [rbp-50h]
  __int32 v20; // [rsp-48h] [rbp-48h]
  char v21; // [rsp-41h] [rbp-41h]

  if ( src != a2 )
  {
    v4 = (const __m128i *)(src + 64);
    if ( src + 64 != a2 )
    {
      for ( i = a3((__m128i *)v4, (const __m128i *)src); ; i = a3((__m128i *)v4, (const __m128i *)src) )
      {
        if ( i )
        {
          v7 = v4->m128i_i64[0];
          v8 = v4->m128i_i64[1];
          v9 = v4[1].m128i_i64[0];
          v10 = v4[1].m128i_i64[1];
          v11 = v4[2].m128i_i32[0];
          v21 = v4[2].m128i_i8[8];
          v12 = v4[2].m128i_i8[4];
          v13 = v4[3].m128i_i64[1];
          v19 = v4[3].m128i_i64[0];
          if ( src != (char *)v4 )
          {
            v15 = v4[1].m128i_i64[0];
            v16 = v4->m128i_i64[0];
            v20 = v4[2].m128i_i32[0];
            v17 = v4[1].m128i_i64[1];
            v18 = v4->m128i_i64[1];
            memmove(src + 64, src, (char *)v4 - src);
            v9 = v15;
            v7 = v16;
            v11 = v20;
            v10 = v17;
            v8 = v18;
          }
          *(_QWORD *)src = v7;
          v4 += 4;
          *((_QWORD *)src + 1) = v8;
          src[40] = v21;
          *((_QWORD *)src + 2) = v9;
          *((_QWORD *)src + 3) = v10;
          *((_DWORD *)src + 8) = v11;
          src[36] = v12;
          *((_QWORD *)src + 6) = v19;
          *((_QWORD *)src + 7) = v13;
          if ( a2 == (char *)v4 )
            return;
        }
        else
        {
          v14 = v4;
          v4 += 4;
          sub_9820C0(v14, a3);
          if ( a2 == (char *)v4 )
            return;
        }
      }
    }
  }
}
