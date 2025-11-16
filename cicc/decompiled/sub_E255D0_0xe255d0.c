// Function: sub_E255D0
// Address: 0xe255d0
//
__int64 __fastcall sub_E255D0(__int64 a1, size_t *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx

  v2 = sub_E25570(a1, a2, 2);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  v3 = v2;
  result = sub_E263F0(a1, a2, v2);
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  if ( *(_DWORD *)(v3 + 8) == 11 )
  {
    v5 = *(_QWORD *)(result + 16);
    v6 = *(_QWORD *)(v5 + 24);
    if ( v6 > 1 )
    {
      *(_QWORD *)(v3 + 24) = *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8 * v6 - 16);
      return result;
    }
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return result;
}
