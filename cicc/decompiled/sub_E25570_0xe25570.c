// Function: sub_E25570
// Address: 0xe25570
//
unsigned __int64 __fastcall sub_E25570(__int64 a1, size_t *a2, char a3)
{
  char *v3; // rdx
  unsigned __int64 v4; // rax
  _WORD *v5; // r8
  int v6; // r9d

  if ( !*a2 )
    return sub_E22500(a1, (__int64 *)a2, (a3 & 2) != 0);
  v5 = (_WORD *)a2[1];
  v6 = *(char *)v5;
  if ( (unsigned int)(v6 - 48) <= 9 )
  {
    v3 = (char *)a2[1];
    v4 = *v3 - 48;
    if ( *(_QWORD *)(a1 + 192) <= v4 )
      JUMPOUT(0xE21CB0);
    --*a2;
    a2[1] = (size_t)(v3 + 1);
    return *(_QWORD *)(a1 + 8 * v4 + 112);
  }
  else if ( *a2 != 1 && *v5 == 9279 )
  {
    return sub_E24FC0(a1, a2, a3);
  }
  else
  {
    if ( (_BYTE)v6 != 63 )
      return sub_E22500(a1, (__int64 *)a2, (a3 & 2) != 0);
    return sub_E22900(a1, a2);
  }
}
