// Function: sub_2216880
// Address: 0x2216880
//
__int64 *__fastcall sub_2216880(__int64 *a1, unsigned __int64 a2, size_t a3)
{
  unsigned __int64 v4; // rax
  __int64 v6; // rbx
  const wchar_t *v7; // rbp
  unsigned __int64 v8; // rbx
  wchar_t *v9; // rdi
  unsigned __int64 v11; // rbp
  bool v12; // cc

  if ( !a3 )
    return a1;
  v4 = *a1;
  v6 = *(_QWORD *)(*a1 - 24);
  if ( a3 > 0xFFFFFFFFFFFFFFELL - v6 )
    sub_4262D8((__int64)"basic_string::append");
  v7 = (const wchar_t *)a2;
  v8 = a3 + v6;
  if ( v8 > *(_QWORD *)(v4 - 16) || (v12 = *(_DWORD *)(v4 - 8) <= 0, v4 = *a1, !v12) )
  {
    if ( v4 <= a2 && a2 <= v4 + 4LL * *(_QWORD *)(v4 - 24) )
    {
      v11 = a2 - v4;
      sub_2216730(a1, v8);
      v4 = *a1;
      v7 = (const wchar_t *)(*a1 + v11);
      v9 = (wchar_t *)(*a1 + 4LL * *(_QWORD *)(*a1 - 24));
      if ( a3 != 1 )
        goto LABEL_8;
      goto LABEL_13;
    }
    sub_2216730(a1, v8);
    v4 = *a1;
  }
  v9 = (wchar_t *)(v4 + 4LL * *(_QWORD *)(v4 - 24));
  if ( a3 != 1 )
  {
LABEL_8:
    wmemcpy(v9, v7, a3);
    v4 = *a1;
    goto LABEL_9;
  }
LABEL_13:
  *v9 = *v7;
LABEL_9:
  if ( (_UNKNOWN *)(v4 - 24) != &unk_4FD67E0 )
  {
    *(_DWORD *)(v4 - 8) = 0;
    *(_QWORD *)(v4 - 24) = v8;
    *(_DWORD *)(v4 + 4 * v8) = 0;
  }
  return a1;
}
