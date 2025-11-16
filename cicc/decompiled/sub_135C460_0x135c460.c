// Function: sub_135C460
// Address: 0x135c460
//
__int64 __fastcall sub_135C460(__int64 a1, __int64 a2, unsigned __int64 a3, const __m128i *a4, char a5)
{
  unsigned __int64 v6; // r9

  v6 = sub_135BF60(a1, a2, a3, a4);
  *(_BYTE *)(v6 + 67) = *(_BYTE *)(v6 + 67) & 0xCF | (16 * ((a5 | (*(_BYTE *)(v6 + 67) >> 4)) & 3));
  if ( *(_QWORD *)(a1 + 64) || *(_DWORD *)(a1 + 56) <= (unsigned int)dword_4F97B80 )
    return v6;
  else
    return sub_135AD00((_QWORD *)a1);
}
