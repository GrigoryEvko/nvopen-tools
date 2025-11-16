// Function: sub_622C70
// Address: 0x622c70
//
__int64 sub_622C70()
{
  __int64 v0; // r14
  __m128i *v1; // r12
  unsigned __int16 *v2; // rbx
  __int64 result; // rax
  int v4; // r8d
  int v5; // [rsp+Ch] [rbp-64h]
  int v6; // [rsp+20h] [rbp-50h] BYREF
  _BOOL4 v7; // [rsp+24h] [rbp-4Ch] BYREF
  __int64 v8; // [rsp+28h] [rbp-48h] BYREF
  __int16 v9[32]; // [rsp+30h] [rbp-40h] BYREF

  v0 = 0;
  v1 = (__m128i *)&unk_4F066C0;
  v2 = (unsigned __int16 *)&unk_4F067A0;
  sub_622920(0, &v8, &v6);
  while ( 1 )
  {
    v4 = v8 * dword_4F06BA0;
    if ( byte_4B6DF90[v0] )
    {
      v5 = v8 * dword_4F06BA0;
      sub_621EE0(v1->m128i_i16, v4 - 1);
      sub_620D80(v9, 1);
      *(__m128i *)v2 = _mm_loadu_si128(v1);
      sub_621270(v2, v9, 0, &v7);
      result = sub_6215A0((__int16 *)v2, v5);
    }
    else
    {
      sub_621EE0(v1->m128i_i16, v4);
      result = (__int64)sub_620D80(v2, 0);
    }
    ++v0;
    ++v1;
    v2 += 8;
    if ( v0 == 13 )
      break;
    sub_622920((unsigned int)v0, &v8, &v6);
  }
  return result;
}
