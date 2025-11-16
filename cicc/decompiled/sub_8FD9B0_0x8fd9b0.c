// Function: sub_8FD9B0
// Address: 0x8fd9b0
//
void __fastcall sub_8FD9B0(__m128i **a1, __int64 a2)
{
  __m128i *v3; // rdi

  v3 = a1[1];
  if ( v3 == a1[2] )
  {
    sub_8FD760(a1, v3, a2);
  }
  else
  {
    if ( v3 )
    {
      v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
      sub_8FC5C0(v3->m128i_i64, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
      v3 = a1[1];
    }
    a1[1] = v3 + 2;
  }
}
