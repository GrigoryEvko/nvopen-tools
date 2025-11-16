// Function: sub_1489BA0
// Address: 0x1489ba0
//
__int64 __fastcall sub_1489BA0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 result; // rax

  result = sub_1481140(*(_QWORD *)a1, 0x26u, a2, a3);
  if ( !(_BYTE)result )
    return sub_1489690(
             *(_QWORD *)a1,
             38,
             a2,
             a3,
             **(_QWORD **)(a1 + 8),
             **(_QWORD **)(a1 + 16),
             a4,
             a5,
             **(_DWORD **)(a1 + 24) + 1);
  return result;
}
