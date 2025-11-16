// Function: sub_7FAF20
// Address: 0x7faf20
//
void __fastcall sub_7FAF20(const __m128i *a1)
{
  __int64 i; // rdi
  __int64 v2; // rax
  __m128i *v3[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( unk_4D03EB0 )
  {
    if ( a1[2].m128i_i8[8] != 11 )
      sub_732CD0(a1, v3);
    sub_7E1740((__int64)a1, (__int64)v3);
    for ( i = unk_4D03EB0; unk_4D03EB0; i = unk_4D03EB0 )
    {
      v2 = *(_QWORD *)(i + 16);
      *(_QWORD *)(i + 16) = 0;
      unk_4D03EB0 = v2;
      sub_7E6810(i, (__int64)v3, 1);
    }
  }
}
