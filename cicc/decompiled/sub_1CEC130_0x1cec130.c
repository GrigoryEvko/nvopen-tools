// Function: sub_1CEC130
// Address: 0x1cec130
//
_QWORD *__fastcall sub_1CEC130(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  const char **v3; // r15
  _QWORD *v4; // r12
  const char *v5; // rdx
  size_t v6; // rcx
  size_t v7; // rax
  const char *v9; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
LABEL_3:
  v3 = (const char **)qword_4FC0600;
  v4 = sub_1648700(v2);
  if ( *((_BYTE *)v4 + 16) != 78 )
    v4 = 0;
  while ( 1 )
  {
    if ( v4 )
    {
      v5 = *v3;
      v6 = 0;
      if ( *v3 )
      {
        v9 = *v3;
        v7 = strlen(*v3);
        v5 = v9;
        v6 = v7;
      }
      if ( sub_1CEBF90(a1, (__int64)v4, v5, v6) )
        return v4;
    }
    if ( ++v3 == &qword_4FC0600[3] )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v2 )
        goto LABEL_3;
      return 0;
    }
  }
}
