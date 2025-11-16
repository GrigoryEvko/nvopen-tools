// Function: sub_CB07C0
// Address: 0xcb07c0
//
__int64 __fastcall sub_CB07C0(__int64 a1, const char *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r13
  size_t v4; // rax

  v2 = *(unsigned __int8 *)(a1 + 680);
  if ( (_BYTE)v2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 672);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v3 + 32LL) - 1) > 1 )
    return 0;
  if ( a2 )
  {
    v4 = strlen(a2);
    if ( v4 != *(_QWORD *)(v3 + 16) )
      return v2;
    if ( v4 && memcmp(*(const void **)(v3 + 8), a2, v4) )
      return 0;
  }
  else if ( *(_QWORD *)(v3 + 16) )
  {
    return v2;
  }
  *(_BYTE *)(a1 + 680) = 1;
  return 1;
}
