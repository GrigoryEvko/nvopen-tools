// Function: sub_73F6B0
// Address: 0x73f6b0
//
_QWORD *__fastcall sub_73F6B0(const __m128i *a1, __int64 a2)
{
  const __m128i *v2; // rbx
  _QWORD *v3; // r13
  _QWORD *i; // r12
  _QWORD *v5; // rax
  __int64 v6; // rcx
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v10; // [rsp+0h] [rbp-40h] BYREF
  int v11; // [rsp+8h] [rbp-38h]

  v10 = 0;
  v11 = 0;
  if ( !a1 )
    return 0;
  v2 = a1;
  v3 = sub_73A9D0(a1, a2, (__int64)&v10);
  for ( i = v3; ; i = v5 )
  {
    v2 = (const __m128i *)v2[1].m128i_i64[0];
    if ( !v2 )
      break;
    v5 = sub_73A9D0(v2, (unsigned int)a2, (__int64)&v10);
    if ( v3 )
      i[2] = v5;
    else
      v3 = v5;
  }
  v6 = (__int64)v10;
  if ( v10 )
  {
    v7 = v10;
    do
    {
      v8 = v7;
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
    *v8 = qword_4F07AD8;
    qword_4F07AD8 = v6;
  }
  return v3;
}
