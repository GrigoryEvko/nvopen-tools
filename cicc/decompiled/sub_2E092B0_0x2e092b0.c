// Function: sub_2E092B0
// Address: 0x2e092b0
//
void __fastcall sub_2E092B0(__int64 *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4, __m128i *a5, __int64 a6)
{
  __int64 v6; // rdi

  v6 = *a1;
  if ( v6 )
  {
    if ( *(_QWORD *)(v6 + 104) )
      sub_2E06810(v6, a2, a3, a4, a5, a6);
    else
      nullsub_2023();
  }
}
