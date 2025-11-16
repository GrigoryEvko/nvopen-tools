// Function: sub_3400EC0
// Address: 0x3400ec0
//
unsigned __int8 *__fastcall sub_3400EC0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __m128i a6)
{
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
    return sub_3400E40(a1, *(_QWORD *)a2, a3, a4, a5, a6);
  else
    return sub_3400E40(a1, **(_QWORD **)a2, a3, a4, a5, a6);
}
