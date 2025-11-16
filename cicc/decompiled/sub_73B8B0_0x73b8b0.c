// Function: sub_73B8B0
// Address: 0x73b8b0
//
_QWORD *__fastcall sub_73B8B0(const __m128i *a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  _QWORD *v4; // r8
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v8; // [rsp+0h] [rbp-10h] BYREF
  int v9; // [rsp+8h] [rbp-8h]

  v8 = 0;
  v9 = 0;
  v2 = sub_73A9D0(a1, a2, (__int64)&v8);
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
