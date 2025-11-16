// Function: sub_BA9430
// Address: 0xba9430
//
void __fastcall sub_BA9430(__int64 **a1, unsigned int a2, const void *a3, size_t a4, unsigned __int8 *a5)
{
  unsigned int v6; // ebx
  __int64 v7; // r13
  int v8; // r14d
  __int8 *v9; // rax
  const void *v10; // rax
  __int64 v11; // rdx
  __m128i *v12; // r15
  unsigned __int8 v13; // al

  v6 = 0;
  v7 = sub_BA92C0((__int64)a1);
  v8 = sub_B91A00(v7);
  if ( v8 )
  {
    while ( 1 )
    {
      v12 = (__m128i *)sub_B91A10(v7, v6);
      v13 = v12[-1].m128i_u8[0];
      v9 = (v13 & 2) != 0 ? (__int8 *)v12[-2].m128i_i64[0] : &v12->m128i_i8[-16 - 8LL * ((v13 >> 2) & 0xF)];
      v10 = (const void *)sub_B91420(*((_QWORD *)v9 + 1));
      if ( a4 == v11 && (!a4 || !memcmp(v10, a3, a4)) )
        break;
      if ( v8 == ++v6 )
        goto LABEL_10;
    }
    sub_BA6610(v12, 2u, a5);
  }
  else
  {
LABEL_10:
    sub_BA92F0(a1, a2, a3, a4, (__int64)a5);
  }
}
