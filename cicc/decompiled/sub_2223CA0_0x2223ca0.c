// Function: sub_2223CA0
// Address: 0x2223ca0
//
_QWORD *__fastcall sub_2223CA0(_QWORD *a1, __int64 a2)
{
  const wchar_t *v2; // rbp

  v2 = *(const wchar_t **)(*(_QWORD *)(a2 + 16) + 64LL);
  *a1 = a1 + 2;
  if ( v2 )
    wcslen(v2);
  sub_22520E0(a1);
  return a1;
}
