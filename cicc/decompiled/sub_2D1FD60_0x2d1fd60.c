// Function: sub_2D1FD60
// Address: 0x2d1fd60
//
_BYTE *__fastcall sub_2D1FD60(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rax
  __int64 v4; // rax
  _BYTE *v5; // r12
  const char **v6; // r13
  const char *v7; // r15
  size_t v8; // rcx

  if ( !sub_D97040(*a1, *(_QWORD *)(a2 + 8)) )
    return 0;
  v2 = *a1;
  v3 = sub_DD8400(*a1, a2);
  v4 = sub_D97190(v2, (__int64)v3);
  if ( *(_WORD *)(v4 + 24) != 15 )
    return 0;
  v5 = *(_BYTE **)(v4 - 8);
  v6 = (const char **)&unk_50164D0;
  if ( *v5 != 85 )
    v5 = 0;
  while ( 1 )
  {
    if ( v5 )
    {
      v7 = *v6;
      v8 = 0;
      if ( *v6 )
        v8 = strlen(*v6);
      if ( sub_2D1FC80(a1, (__int64)v5, v7, v8) )
        break;
    }
    if ( ++v6 == (const char **)&dword_50164E8 )
      return 0;
  }
  return v5;
}
