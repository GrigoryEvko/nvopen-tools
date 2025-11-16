// Function: sub_169ED90
// Address: 0x169ed90
//
void *__fastcall sub_169ED90(void **a1, void **a2)
{
  void **v2; // r13
  void *result; // rax
  void *v4; // r14
  void *v5; // rbx
  _QWORD *v6; // r15
  __int64 v7; // rsi
  _QWORD *v8; // rbx

  v2 = a2;
  result = sub_16982C0();
  v4 = *a2;
  v5 = result;
  if ( *a1 != result )
  {
    if ( result != v4 )
      return (void *)sub_16983E0((__int64)a1, (__int64)a2);
LABEL_4:
    if ( a1 != a2 )
    {
      sub_127D120(a1);
      if ( v5 != *a2 )
        return (void *)sub_1698450((__int64)a1, (__int64)a2);
      return sub_169C7E0(a1, a2);
    }
    return result;
  }
  if ( result != v4 )
    goto LABEL_4;
  if ( a2 != a1 )
  {
    v6 = a1[1];
    if ( v6 )
    {
      v7 = 4LL * *(v6 - 1);
      v8 = &v6[v7];
      while ( v6 != v8 )
      {
        v8 -= 4;
        if ( (void *)v8[1] == v4 )
          sub_169DEB0(v8 + 2);
        else
          sub_1698460((__int64)(v8 + 1));
      }
      j_j_j___libc_free_0_0(v6 - 1);
    }
    a2 = v2;
    return sub_169C7E0(a1, a2);
  }
  return result;
}
