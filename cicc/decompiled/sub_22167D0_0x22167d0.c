// Function: sub_22167D0
// Address: 0x22167d0
//
__int64 *__fastcall sub_22167D0(__int64 *a1, const wchar_t **a2)
{
  size_t v3; // r13
  size_t v4; // rbp
  __int64 v5; // rax
  const wchar_t *v6; // rsi
  wchar_t *v7; // rdi

  v3 = *((_QWORD *)*a2 - 3);
  if ( v3 )
  {
    v4 = v3 + *(_QWORD *)(*a1 - 24);
    if ( v4 > *(_QWORD *)(*a1 - 16) || *(int *)(*a1 - 8) > 0 )
      sub_2216730(a1, v3 + *(_QWORD *)(*a1 - 24));
    v5 = *a1;
    v6 = *a2;
    v7 = (wchar_t *)(*a1 + 4LL * *(_QWORD *)(*a1 - 24));
    if ( v3 == 1 )
    {
      *v7 = *v6;
    }
    else
    {
      wmemcpy(v7, v6, v3);
      v5 = *a1;
    }
    if ( (_UNKNOWN *)(v5 - 24) != &unk_4FD67E0 )
    {
      *(_DWORD *)(v5 - 8) = 0;
      *(_QWORD *)(v5 - 24) = v4;
      *(_DWORD *)(v5 + 4 * v4) = 0;
    }
  }
  return a1;
}
