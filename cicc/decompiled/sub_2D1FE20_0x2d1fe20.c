// Function: sub_2D1FE20
// Address: 0x2d1fe20
//
_BYTE *__fastcall sub_2D1FE20(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // r15
  const char **v4; // rbx
  const char *v5; // rdx
  size_t v6; // rcx
  size_t v7; // rax
  const char *v9; // [rsp+8h] [rbp-38h]

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 )
    return 0;
LABEL_3:
  v3 = *(_BYTE **)(v2 + 24);
  v4 = (const char **)qword_50164B0;
  if ( *v3 != 85 )
    v3 = 0;
  while ( 1 )
  {
    if ( v3 )
    {
      v5 = *v4;
      v6 = 0;
      if ( *v4 )
      {
        v9 = *v4;
        v7 = strlen(*v4);
        v5 = v9;
        v6 = v7;
      }
      if ( sub_2D1FC80(a1, (__int64)v3, v5, v6) )
        return v3;
    }
    if ( ++v4 == &qword_50164B0[3] )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v2 )
        goto LABEL_3;
      return 0;
    }
  }
}
