// Function: sub_FD9620
// Address: 0xfd9620
//
__int64 __fastcall sub_FD9620(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, const __m128i a7)
{
  __int64 v7; // r8

  v7 = sub_FD88D0(a1, &a7, a3, a4, a5, a6);
  *(_BYTE *)(v7 + 67) = *(_BYTE *)(v7 + 67) & 0xCF | (16 * ((a2 | (*(_BYTE *)(v7 + 67) >> 4)) & 3));
  if ( !*(_QWORD *)(a1 + 64) && *(_DWORD *)(a1 + 56) > (unsigned int)qword_4F8D968 )
    return sub_FD8300(a1);
  return v7;
}
