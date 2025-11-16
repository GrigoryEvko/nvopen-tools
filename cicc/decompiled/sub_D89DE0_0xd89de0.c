// Function: sub_D89DE0
// Address: 0xd89de0
//
__int64 (__fastcall *__fastcall sub_D89DE0(__int64 a1, __int64 a2))(__int64, __int64, __int64)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r13
  _QWORD *v8; // rdi
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  v3 = *(_QWORD *)(a1 + 48);
  if ( !v3 )
    goto LABEL_8;
  v4 = *(_QWORD *)(v3 + 160);
  while ( v4 )
  {
    sub_D85A50(*(_QWORD *)(v4 + 24));
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 16);
    a2 = 40;
    j_j___libc_free_0(v5, 40);
  }
  if ( *(_BYTE *)(v3 + 76) )
  {
    v6 = *(_QWORD *)(v3 + 16);
    if ( v6 )
      goto LABEL_6;
  }
  else
  {
    _libc_free(*(_QWORD *)(v3 + 56), a2);
    v6 = *(_QWORD *)(v3 + 16);
    while ( v6 )
    {
LABEL_6:
      v7 = v6;
      sub_D86030(*(_QWORD **)(v6 + 24));
      v8 = *(_QWORD **)(v6 + 104);
      v6 = *(_QWORD *)(v6 + 16);
      sub_D85F30(v8);
      sub_D85E30(*(_QWORD **)(v7 + 56));
      j_j___libc_free_0(v7, 144);
    }
  }
  j_j___libc_free_0(v3, 192);
LABEL_8:
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  return result;
}
