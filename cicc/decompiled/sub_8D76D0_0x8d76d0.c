// Function: sub_8D76D0
// Address: 0x8d76d0
//
__int64 __fastcall sub_8D76D0(__int64 a1)
{
  __int64 v1; // rbx
  __m128i *v2; // rdi

  if ( !dword_4D048B8 )
    return 1;
  if ( *(_BYTE *)(a1 + 140) == 13 )
  {
    v1 = *(_QWORD *)(sub_8D4870(a1) + 168);
    v2 = *(__m128i **)(v1 + 56);
    if ( v2 )
      goto LABEL_4;
  }
  else
  {
    v1 = *(_QWORD *)(a1 + 168);
    v2 = *(__m128i **)(v1 + 56);
    if ( v2 )
    {
LABEL_4:
      if ( (v2->m128i_i8[0] & 2) == 0 )
        return sub_8D7650(v2);
      sub_5F80E0(*(_QWORD *)(v1 + 8));
      v2 = *(__m128i **)(v1 + 56);
      if ( v2 )
        return sub_8D7650(v2);
    }
  }
  return 0;
}
