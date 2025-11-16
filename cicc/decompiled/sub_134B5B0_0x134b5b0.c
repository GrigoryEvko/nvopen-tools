// Function: sub_134B5B0
// Address: 0x134b5b0
//
__int64 __fastcall sub_134B5B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // rcx
  __int64 result; // rax
  __int64 v5; // rdx

  v2 = a2;
  v3 = sub_134B450(a2);
  result = a1 + 8 * v3;
  if ( !*(_QWORD *)(result + 4232) )
    *(_QWORD *)(a1 + 8 * (v3 >> 6) + 5256) |= 1LL << (v3 & 0x3F);
  *(_QWORD *)(a2 + 64) = a2;
  *(_QWORD *)(a2 + 72) = a2;
  v5 = *(_QWORD *)(result + 4232);
  if ( v5 )
  {
    *(_QWORD *)(a2 + 64) = *(_QWORD *)(v5 + 72);
    *(_QWORD *)(*(_QWORD *)(result + 4232) + 72LL) = a2;
    *(_QWORD *)(a2 + 72) = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 64LL);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(result + 4232) + 72LL) + 64LL) = *(_QWORD *)(result + 4232);
    *(_QWORD *)(*(_QWORD *)(a2 + 72) + 64LL) = a2;
    v2 = *(_QWORD *)(a2 + 64);
  }
  *(_QWORD *)(result + 4232) = v2;
  return result;
}
