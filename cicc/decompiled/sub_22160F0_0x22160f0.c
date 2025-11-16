// Function: sub_22160F0
// Address: 0x22160f0
//
void *__fastcall sub_22160F0(size_t n, wchar_t c)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r13

  if ( !n )
    return &unk_4FD67F8;
  v4 = sub_2216040(n, 0);
  v5 = v4;
  v2 = v4 + 24;
  if ( n == 1 )
    *(_DWORD *)(v4 + 24) = c;
  else
    wmemset((wchar_t *)(v4 + 24), c, n);
  if ( (_UNKNOWN *)v5 != &unk_4FD67E0 )
  {
    *(_DWORD *)(v5 + 16) = 0;
    *(_QWORD *)v5 = n;
    *(_DWORD *)(v5 + 4 * n + 24) = 0;
  }
  return (void *)v2;
}
