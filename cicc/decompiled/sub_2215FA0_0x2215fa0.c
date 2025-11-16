// Function: sub_2215FA0
// Address: 0x2215fa0
//
__int64 __fastcall sub_2215FA0(__int64 *a1, wchar_t *a2, size_t a3, unsigned __int64 a4)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  size_t v6; // r12
  const wchar_t *v8; // rsi

  v4 = *a1;
  v5 = *(_QWORD *)(*a1 - 24);
  if ( a4 > v5 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::copy");
  v6 = v5 - a4;
  if ( v6 > a3 )
    v6 = a3;
  if ( !v6 )
    return v6;
  v8 = (const wchar_t *)(v4 + 4 * a4);
  if ( v6 != 1 )
  {
    wmemcpy(a2, v8, v6);
    return v6;
  }
  *a2 = *v8;
  return 1;
}
