// Function: sub_773170
// Address: 0x773170
//
__int64 __fastcall sub_773170(__int64 a1, __int64 a2, __int64 a3, _BYTE **a4, __m128i *a5)
{
  _BYTE *v6; // rdi
  __int64 v8; // rdx
  char v9; // al

  v6 = *a4;
  if ( **a4 == 48 )
  {
    v8 = *((_QWORD *)v6 + 1);
    v9 = *(_BYTE *)(v8 + 8);
    if ( v9 == 1 )
    {
      *v6 = 2;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v8 + 32);
    }
    else if ( v9 == 2 )
    {
      *v6 = 59;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v8 + 32);
    }
    else
    {
      if ( v9 )
        sub_721090();
      *v6 = 6;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v8 + 32);
    }
  }
  if ( sub_773040(v6) )
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08280);
  else
    *a5 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
  return 1;
}
