// Function: sub_18D1D70
// Address: 0x18d1d70
//
__int64 *__fastcall sub_18D1D70(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int16 v3; // dx
  __int64 *result; // rax

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 == *(_QWORD *)(a1 + 16) )
    return sub_18D1A20((__int64 *)a1, *(char **)(a1 + 8), (__int64 *)a2);
  if ( v2 )
  {
    *(_QWORD *)v2 = *(_QWORD *)a2;
    v3 = *(_WORD *)(a2 + 8);
    *(_BYTE *)(v2 + 10) = *(_BYTE *)(a2 + 10);
    *(_WORD *)(v2 + 8) = v3;
    *(_WORD *)(v2 + 16) = *(_WORD *)(a2 + 16);
    *(_QWORD *)(v2 + 24) = *(_QWORD *)(a2 + 24);
    sub_16CCEE0((_QWORD *)(v2 + 32), v2 + 72, 2, a2 + 32);
    sub_16CCEE0((_QWORD *)(v2 + 88), v2 + 128, 2, a2 + 88);
    result = (__int64 *)*(unsigned __int8 *)(a2 + 144);
    *(_BYTE *)(v2 + 144) = (_BYTE)result;
    v2 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v2 + 152;
  return result;
}
