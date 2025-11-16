// Function: sub_1AC64D0
// Address: 0x1ac64d0
//
__int64 *__fastcall sub_1AC64D0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *result; // rax
  __int64 v4; // rdx

  result = a3;
  if ( *(_BYTE *)(a2 + 16) == 5 && a3 && *(_WORD *)(a2 + 18) == 47 )
  {
    v4 = **(_QWORD **)(*(_QWORD *)a2 + 16LL);
    if ( *(_BYTE *)(v4 + 8) == 12 )
      return (__int64 *)sub_14D66F0(result, **(_QWORD **)(v4 + 16), *(_QWORD *)(a1 + 640));
  }
  return result;
}
