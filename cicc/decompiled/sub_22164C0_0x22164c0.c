// Function: sub_22164C0
// Address: 0x22164c0
//
const wchar_t **__fastcall sub_22164C0(const wchar_t **a1, size_t a2, __int64 a3, const wchar_t *a4, size_t a5)
{
  wchar_t *v8; // rdi

  sub_22161B0(a1, a2, a3, a5);
  if ( !a5 )
    return a1;
  v8 = (wchar_t *)&(*a1)[a2];
  if ( a5 != 1 )
  {
    wmemcpy(v8, a4, a5);
    return a1;
  }
  *v8 = *a4;
  return a1;
}
