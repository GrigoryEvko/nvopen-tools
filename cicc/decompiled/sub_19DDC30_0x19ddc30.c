// Function: sub_19DDC30
// Address: 0x19ddc30
//
__int64 *__fastcall sub_19DDC30(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 *result; // rax

  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_QWORD *)(a2 - 24);
  v6 = sub_146F1B0(*(_QWORD *)(a1 + 24), a2);
  if ( sub_14560B0(v6) )
    return 0;
  result = sub_19DD950(a1, v4, v5, a2, a3, a4);
  if ( !result )
    return sub_19DD950(a1, v5, v4, a2, a3, a4);
  return result;
}
