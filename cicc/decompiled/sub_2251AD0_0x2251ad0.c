// Function: sub_2251AD0
// Address: 0x2251ad0
//
__int64 __fastcall sub_2251AD0(__int64 a1, size_t a2, __int64 a3, size_t a4, wchar_t a5)
{
  __int64 v6; // rcx
  const wchar_t *v7; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rsi
  size_t v13; // rcx
  const wchar_t *v14; // rsi
  const wchar_t *v15; // r8
  wchar_t *v16; // rdi
  wchar_t *v17; // rdi

  v6 = *(_QWORD *)(a1 + 8);
  if ( a4 > a3 + 0xFFFFFFFFFFFFFFFLL - v6 )
    sub_4262D8((__int64)"basic_string::_M_replace_aux");
  v7 = *(const wchar_t **)a1;
  v11 = v6 + a4 - a3;
  if ( *(_QWORD *)a1 == a1 + 16 )
    v12 = 3;
  else
    v12 = *(_QWORD *)(a1 + 16);
  if ( v12 < v11 )
  {
    sub_2251880((const wchar_t **)a1, a2, a3, 0, a4);
    v7 = *(const wchar_t **)a1;
    if ( !a4 )
      goto LABEL_12;
  }
  else
  {
    v13 = v6 - (a3 + a2);
    if ( !v13 || a3 == a4 )
    {
LABEL_9:
      if ( !a4 )
        goto LABEL_12;
      goto LABEL_10;
    }
    v14 = &v7[a2];
    v15 = &v14[a3];
    v16 = (wchar_t *)&v14[a4];
    if ( v13 != 1 )
    {
      wmemmove(v16, v15, v13);
      v7 = *(const wchar_t **)a1;
      goto LABEL_9;
    }
    *v16 = *v15;
    if ( !a4 )
      goto LABEL_12;
  }
LABEL_10:
  v17 = (wchar_t *)&v7[a2];
  if ( a4 == 1 )
  {
    *v17 = a5;
  }
  else
  {
    wmemset(v17, a5, a4);
    v7 = *(const wchar_t **)a1;
  }
LABEL_12:
  *(_QWORD *)(a1 + 8) = v11;
  v7[v11] = 0;
  return a1;
}
