// Function: sub_740B80
// Address: 0x740b80
//
__m128i *__fastcall sub_740B80(__int64 a1, unsigned int a2)
{
  __m128i *v2; // rax
  __int64 v3; // rcx
  __m128i *v4; // r8
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v8; // [rsp+0h] [rbp-10h] BYREF
  int v9; // [rsp+8h] [rbp-8h]

  v8 = 0;
  v9 = 0;
  v2 = sub_73F780(a1, a2, &v8);
  v3 = (__int64)v8;
  v4 = v2;
  if ( v8 )
  {
    v5 = v8;
    do
    {
      v6 = v5;
      v5 = (_QWORD *)*v5;
    }
    while ( v5 );
    *v6 = qword_4F07AD8;
    qword_4F07AD8 = v3;
  }
  return v4;
}
