// Function: sub_740190
// Address: 0x740190
//
__m128i *__fastcall sub_740190(__int64 a1, __m128i *a2, unsigned int a3)
{
  __m128i *v3; // rax
  __int64 v4; // rcx
  __m128i *v5; // r8
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *v9; // [rsp+0h] [rbp-10h] BYREF
  int v10; // [rsp+8h] [rbp-8h]

  v9 = 0;
  v10 = 0;
  v3 = sub_73FC90(a1, a2, a3, &v9);
  v4 = (__int64)v9;
  v5 = v3;
  if ( v9 )
  {
    v6 = v9;
    do
    {
      v7 = v6;
      v6 = (_QWORD *)*v6;
    }
    while ( v6 );
    *v7 = qword_4F07AD8;
    qword_4F07AD8 = v4;
  }
  return v5;
}
