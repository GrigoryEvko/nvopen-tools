// Function: sub_E253C0
// Address: 0xe253c0
//
__int64 __fastcall sub_E253C0(__int64 a1, __int64 *a2, char a3)
{
  char *v3; // rdx
  unsigned __int64 v4; // rax
  _WORD *v5; // r8

  if ( !*a2 )
    return sub_E22500(a1, a2, a3);
  v5 = (_WORD *)a2[1];
  if ( (unsigned int)(*(char *)v5 - 48) > 9 )
  {
    if ( *a2 != 1 && *v5 == 9279 )
      return sub_E24FC0(a1, (size_t *)a2, 1);
    return sub_E22500(a1, a2, a3);
  }
  v3 = (char *)a2[1];
  v4 = *v3 - 48;
  if ( *(_QWORD *)(a1 + 192) <= v4 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  else
  {
    --*a2;
    a2[1] = (__int64)(v3 + 1);
    return *(_QWORD *)(a1 + 8 * v4 + 112);
  }
}
