// Function: sub_21C72B0
// Address: 0x21c72b0
//
__int64 __fastcall sub_21C72B0(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8)
{
  __int64 v8; // rdx
  _QWORD *v9; // rax

  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( (unsigned int)v9 > 0xFDE )
  {
    if ( (unsigned int)((_DWORD)v9 - 4069) > 2 )
      return 0;
  }
  else if ( (unsigned int)v9 <= 0xFDB )
  {
    return 0;
  }
  return sub_21C5A60(a1, a2, a3, a4, a5, v8, a7, a8);
}
