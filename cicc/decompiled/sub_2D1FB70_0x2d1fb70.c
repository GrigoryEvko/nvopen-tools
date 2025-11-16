// Function: sub_2D1FB70
// Address: 0x2d1fb70
//
char *__fastcall sub_2D1FB70(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rax
  char *v5; // r12
  char v6; // al

  if ( !sub_D97040(*a1, *(_QWORD *)(a2 + 8)) )
    return 0;
  v2 = *a1;
  v3 = sub_DD8400(*a1, a2);
  v4 = sub_D97190(v2, (__int64)v3);
  if ( *(_WORD *)(v4 + 24) != 15 )
    return 0;
  v5 = *(char **)(v4 - 8);
  v6 = *v5;
  if ( (unsigned __int8)*v5 <= 0x1Cu || v6 != 60 && (v6 != 79 || !sub_CEFE10((__int64)v5) && !sub_CEFE40((__int64)v5)) )
    return 0;
  return v5;
}
