// Function: sub_1B3BD80
// Address: 0x1b3bd80
//
__int64 *__fastcall sub_1B3BD80(_QWORD *a1, __int64 **a2, __int64 a3, __int64 *a4, char *a5, size_t a6)
{
  __int64 *result; // rax
  __int64 *v8; // r12
  __int64 *v9; // rdi
  const char *v10; // rax
  size_t v11; // rdx

  result = (__int64 *)&unk_49F6870;
  *a1 = &unk_49F6870;
  a1[1] = a4;
  if ( a3 )
  {
    v8 = *a2;
    v9 = a4;
    if ( *((_BYTE *)*a2 + 16) == 54 )
    {
      if ( a6 )
        return sub_1B3B8C0(v9, *v8, a5, a6);
    }
    else
    {
      v8 = (__int64 *)*(v8 - 6);
      if ( a6 )
        return sub_1B3B8C0(v9, *v8, a5, a6);
    }
    v10 = sub_1649960((__int64)v8);
    v9 = (__int64 *)a1[1];
    a5 = (char *)v10;
    a6 = v11;
    return sub_1B3B8C0(v9, *v8, a5, a6);
  }
  return result;
}
