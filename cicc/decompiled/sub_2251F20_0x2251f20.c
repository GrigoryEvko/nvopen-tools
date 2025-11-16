// Function: sub_2251F20
// Address: 0x2251f20
//
__int64 __fastcall sub_2251F20(__int64 a1, const wchar_t *a2, size_t a3)
{
  __int64 v5; // r9
  const wchar_t *v6; // rax
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rdx
  wchar_t *v9; // rdi

  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(const wchar_t **)a1;
  v7 = v5 + a3;
  if ( *(_QWORD *)a1 == a1 + 16 )
    v8 = 3;
  else
    v8 = *(_QWORD *)(a1 + 16);
  if ( v7 > v8 )
  {
    sub_2251880((const wchar_t **)a1, *(_QWORD *)(a1 + 8), 0, a2, a3);
    v6 = *(const wchar_t **)a1;
  }
  else if ( a3 )
  {
    v9 = (wchar_t *)&v6[v5];
    if ( a3 == 1 )
    {
      *v9 = *a2;
    }
    else
    {
      wmemcpy(v9, a2, a3);
      v6 = *(const wchar_t **)a1;
    }
  }
  *(_QWORD *)(a1 + 8) = v7;
  v6[v7] = 0;
  return a1;
}
