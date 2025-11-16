// Function: sub_1489C20
// Address: 0x1489c20
//
__int64 __fastcall sub_1489C20(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  if ( (unsigned __int8)sub_1481140(*(_QWORD *)(a1 + 8), **(_DWORD **)a1, a2, a3)
    || (unsigned __int8)sub_1479370(
                          *(_QWORD *)(a1 + 8),
                          **(_DWORD **)a1,
                          a2,
                          a3,
                          **(_QWORD **)(a1 + 16),
                          **(_QWORD **)(a1 + 24)) )
  {
    return 1;
  }
  else
  {
    return sub_1489690(
             *(_QWORD *)(a1 + 8),
             **(_DWORD **)a1,
             a2,
             a3,
             **(_QWORD **)(a1 + 16),
             **(_QWORD **)(a1 + 24),
             a4,
             a5,
             **(_DWORD **)(a1 + 32));
  }
}
