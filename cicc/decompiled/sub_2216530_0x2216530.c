// Function: sub_2216530
// Address: 0x2216530
//
const wchar_t **__fastcall sub_2216530(const wchar_t **a1, const wchar_t *a2, size_t a3)
{
  const wchar_t *v4; // rax
  __int64 v5; // rdx
  wchar_t *v8; // rdi
  unsigned __int64 v9; // rax
  size_t v10; // [rsp+8h] [rbp-10h]
  size_t v11; // [rsp+8h] [rbp-10h]

  v4 = *a1;
  v5 = *((_QWORD *)*a1 - 3);
  if ( a3 > 0xFFFFFFFFFFFFFFELL )
    sub_4262D8((__int64)"basic_string::assign");
  if ( v4 > a2 || &v4[v5] < a2 )
    return sub_22164C0(a1, 0, v5, a2, a3);
  if ( *(v4 - 2) > 0 )
  {
    v5 = *((_QWORD *)*a1 - 3);
    return sub_22164C0(a1, 0, v5, a2, a3);
  }
  v8 = (wchar_t *)*a1;
  v9 = a2 - v8;
  if ( a3 <= v9 )
  {
    if ( a3 != 1 )
    {
      if ( a3 )
      {
        v11 = a3;
        wmemcpy(v8, a2, a3);
        v8 = (wchar_t *)*a1;
        a3 = v11;
      }
      goto LABEL_12;
    }
LABEL_18:
    *v8 = *a2;
    goto LABEL_12;
  }
  if ( v9 )
  {
    if ( a3 != 1 )
    {
      if ( a3 )
      {
        v10 = a3;
        wmemmove(v8, a2, a3);
        v8 = (wchar_t *)*a1;
        a3 = v10;
      }
      goto LABEL_12;
    }
    goto LABEL_18;
  }
LABEL_12:
  if ( v8 - 6 != (wchar_t *)&unk_4FD67E0 )
  {
    *(v8 - 2) = 0;
    *((_QWORD *)v8 - 3) = a3;
    v8[a3] = 0;
  }
  return a1;
}
