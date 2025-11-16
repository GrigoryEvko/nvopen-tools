// Function: sub_7FAFA0
// Address: 0x7fafa0
//
void __fastcall sub_7FAFA0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax

  v2 = unk_4D03EB0;
  if ( unk_4D03EB0 )
  {
    if ( *(_QWORD *)(a1 + 16) )
    {
      sub_7FAF20(*(const __m128i **)(a1 + 16));
    }
    else
    {
      do
      {
        v3 = *(_QWORD *)(v2 + 16);
        *(_QWORD *)(v2 + 16) = 0;
        unk_4D03EB0 = v3;
        sub_7E6810(v2, a1, 1);
        v2 = unk_4D03EB0;
      }
      while ( unk_4D03EB0 );
    }
  }
  sub_7E2BD0(a1);
}
