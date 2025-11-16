// Function: sub_E25DD0
// Address: 0xe25dd0
//
__int64 __fastcall sub_E25DD0(__int64 a1, size_t *a2)
{
  size_t v2; // rax
  _BYTE *v3; // rdx
  __int64 result; // rax

  v2 = *a2;
  if ( !*a2 )
    goto LABEL_6;
  v3 = (_BYTE *)a2[1];
  if ( *v3 == 46 )
    return sub_E279E0();
  if ( v2 > 2 && *(_WORD *)v3 == 16191 && v3[2] == 64 )
    return sub_E21710(a1, a2);
  if ( *v3 != 63 )
  {
LABEL_6:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  a2[1] = (size_t)(v3 + 1);
  *a2 = v2 - 1;
  result = sub_E258E0(a1, a2);
  if ( !result )
    return sub_E25660(a1, a2);
  return result;
}
