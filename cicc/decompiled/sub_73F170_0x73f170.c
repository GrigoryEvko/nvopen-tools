// Function: sub_73F170
// Address: 0x73f170
//
_QWORD *__fastcall sub_73F170(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v7; // r13
  _QWORD *result; // rax

  sub_724C70(a2, 7);
  *(_BYTE *)(a2 + 192) |= 2u;
  *(_QWORD *)(a2 + 200) = a1;
  v6 = *a1;
  v7 = *(_QWORD *)(*a1 + 64);
  if ( (*((_BYTE *)a1 + 195) & 1) != 0 )
    sub_894C00(v6, 7, v3, v4, v5);
  result = sub_73F0A0((__m128i *)a1[19], v7);
  *(_QWORD *)(a2 + 128) = result;
  return result;
}
