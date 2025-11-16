// Function: sub_2216980
// Address: 0x2216980
//
__int64 *__fastcall sub_2216980(__int64 *a1, size_t a2, wchar_t a3)
{
  __int64 v4; // rcx
  __int64 v6; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  wchar_t *v9; // rdi

  if ( a2 )
  {
    v4 = *a1;
    v6 = *(_QWORD *)(*a1 - 24);
    if ( a2 > 0xFFFFFFFFFFFFFFELL - v6 )
      sub_4262D8((__int64)"basic_string::append");
    v7 = a2 + v6;
    if ( a2 + v6 > *(_QWORD *)(v4 - 16) || *(int *)(v4 - 8) > 0 )
      sub_2216730(a1, v7);
    v8 = *a1;
    v9 = (wchar_t *)(*a1 + 4LL * *(_QWORD *)(*a1 - 24));
    if ( a2 == 1 )
    {
      *v9 = a3;
    }
    else
    {
      wmemset(v9, a3, a2);
      v8 = *a1;
    }
    if ( (_UNKNOWN *)(v8 - 24) != &unk_4FD67E0 )
    {
      *(_DWORD *)(v8 - 8) = 0;
      *(_QWORD *)(v8 - 24) = v7;
      *(_DWORD *)(v8 + 4 * v7) = 0;
    }
  }
  return a1;
}
