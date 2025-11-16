// Function: sub_7D9EC0
// Address: 0x7d9ec0
//
__int64 __fastcall sub_7D9EC0(_QWORD *a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( (*((_BYTE *)a1 + 25) & 4) != 0
    && *((_BYTE *)a1 + 24) == 1
    && *((_BYTE *)a1 + 56) == 5
    && (unsigned int)sub_8D2600(*a1) )
  {
    a2 = (const __m128i *)a1[9];
    sub_730620((__int64)a1, a2);
  }
  sub_7D9DD0(a1, (__int64)a2, a3, a4, a5, a6);
  return sub_7E7010(a1);
}
