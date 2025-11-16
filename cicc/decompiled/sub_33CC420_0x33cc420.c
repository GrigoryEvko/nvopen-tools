// Function: sub_33CC420
// Address: 0x33cc420
//
__int64 __fastcall sub_33CC420(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  _QWORD *v3; // rbx
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 400);
  *(_QWORD *)(a2 + 16) = a1 + 400;
  v2 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a2 + 8) = v2 | *(_QWORD *)(a2 + 8) & 7LL;
  *(_QWORD *)(v2 + 8) = a2 + 8;
  v3 = *(_QWORD **)(a1 + 768);
  result = *(_QWORD *)(a1 + 400) & 7LL;
  for ( *(_QWORD *)(a1 + 400) = result | (a2 + 8); v3; v3 = (_QWORD *)v3[1] )
    result = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v3 + 32LL))(v3, a2);
  return result;
}
