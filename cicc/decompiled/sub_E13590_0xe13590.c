// Function: sub_E13590
// Address: 0xe13590
//
__int64 *__fastcall sub_E13590(_DWORD *a1, __int64 *a2)
{
  __int64 v3; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  __int64 *result; // rax
  size_t v12; // r13
  const void *v13; // rdx
  const void *v14; // r14
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  char *v17; // rdi
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax

  v3 = a2[1];
  v5 = a2[2];
  v6 = (char *)*a2;
  if ( v3 + 5 > v5 )
  {
    v7 = v3 + 997;
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = v8;
    else
      a2[2] = v7;
    v9 = realloc(v6);
    *a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_19;
    v3 = a2[1];
  }
  v10 = &v6[v3];
  *(_DWORD *)v10 = 979661939;
  v10[4] = 58;
  a2[1] += 5;
  result = (__int64 *)(*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a1 + 56LL))(a1);
  v12 = (size_t)result;
  v14 = v13;
  if ( !result )
    goto LABEL_7;
  v15 = a2[1];
  v16 = a2[2];
  v17 = (char *)*a2;
  if ( v12 + v15 > v16 )
  {
    v18 = v12 + v15 + 992;
    v19 = 2 * v16;
    if ( v18 > v19 )
      a2[2] = v18;
    else
      a2[2] = v19;
    v20 = realloc(v17);
    *a2 = v20;
    v17 = (char *)v20;
    if ( v20 )
    {
      v15 = a2[1];
      goto LABEL_16;
    }
LABEL_19:
    abort();
  }
LABEL_16:
  result = (__int64 *)memcpy(&v17[v15], v14, v12);
  a2[1] += v12;
LABEL_7:
  if ( a1[3] > 1u )
  {
    sub_E12F20(a2, 0x1Du, "<char, std::char_traits<char>");
    if ( a1[3] == 2 )
      sub_E12F20(a2, 0x16u, ", std::allocator<char>");
    return sub_E12F20(a2, 1u, ">");
  }
  return result;
}
