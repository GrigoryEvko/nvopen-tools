// Function: sub_E25660
// Address: 0xe25660
//
__int64 __fastcall sub_E25660(__int64 a1, size_t *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // rdx

  v2 = sub_E255D0(a1, a2);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  v3 = v2;
  result = sub_E29250(a1, a2, v2);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  *(_QWORD *)(result + 16) = v3;
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 16) + 16LL) + 8LL * *(_QWORD *)(*(_QWORD *)(v3 + 16) + 24LL) - 8);
  if ( *(_DWORD *)(v5 + 8) == 9 && !*(_QWORD *)(v5 + 24) )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return result;
}
