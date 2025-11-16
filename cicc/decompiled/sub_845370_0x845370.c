// Function: sub_845370
// Address: 0x845370
//
__int64 __fastcall sub_845370(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 v4; // [rsp-18h] [rbp-18h]

  if ( *(_QWORD *)a3 || (*(_WORD *)(a3 + 16) & 0x101) != 0 )
    return sub_8449E0(a1, a2, a3, 0, 0);
  sub_6FCCE0((__int64)a2, a1, 0, 1, ((*(_BYTE *)(a3 + 17) >> 1) ^ 1) & 1, 0, 0);
  return v4;
}
