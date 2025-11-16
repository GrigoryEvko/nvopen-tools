// Function: sub_740640
// Address: 0x740640
//
void __fastcall sub_740640(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int8 v2; // r15
  unsigned __int64 v3; // r14
  __m128i *v4; // rax
  unsigned __int64 v5; // rax
  unsigned int v6; // [rsp-54h] [rbp-54h]
  unsigned __int64 v7; // [rsp-50h] [rbp-50h]
  const __m128i *v8; // [rsp-40h] [rbp-40h] BYREF

  if ( *(_BYTE *)(a1 + 173) == 2 )
  {
    v1 = *(_QWORD *)(a1 + 184);
    v2 = *(_BYTE *)(a1 + 168) & 7;
    v6 = qword_4F06B40[v2];
    v7 = *(_QWORD *)(a1 + 176);
    v8 = (const __m128i *)sub_724DC0();
    sub_724C70((__int64)v8, 1);
    v8[8].m128i_i64[0] = (__int64)sub_72C330(v2);
    v3 = 0;
    sub_724A80(a1, 10);
    if ( v7 )
    {
      do
      {
        if ( v2 )
        {
          v5 = sub_722AB0((unsigned __int8 *)(v1 + v3), v6);
          sub_620DE0((const __m128i *)v8[11].m128i_i16, v5);
        }
        else
        {
          sub_620D80((const __m128i *)v8[11].m128i_i16, *(char *)(v1 + v3));
        }
        v3 += v6;
        v4 = sub_740630(v8);
        sub_72A690((__int64)v4, a1, 0, 0);
      }
      while ( v7 > v3 );
    }
    sub_724E30((__int64)&v8);
  }
}
