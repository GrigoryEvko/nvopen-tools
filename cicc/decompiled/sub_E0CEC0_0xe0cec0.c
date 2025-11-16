// Function: sub_E0CEC0
// Address: 0xe0cec0
//
__int64 __fastcall sub_E0CEC0(__int64 a1, unsigned __int64 a2, char *a3)
{
  const char *v4; // rax
  const char *v5; // r14
  size_t v6; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( (unsigned __int8)sub_E0CD60(a2, a3, a1, 1, 1u)
    || a2 && *a3 == 95 && (unsigned __int8)sub_E0CD60(a2 - 1, a3 + 1, a1, 0, 1u) )
  {
    return a1;
  }
  v4 = (const char *)sub_E29610(a2, a3, 0, 0, 0);
  v5 = v4;
  if ( v4 )
  {
    v6 = strlen(v4);
    sub_2241130(a1, 0, *(_QWORD *)(a1 + 8), v5, v6);
    _libc_free(v5, 0);
    return a1;
  }
  sub_2241130(a1, 0, *(_QWORD *)(a1 + 8), a3, a2);
  return a1;
}
