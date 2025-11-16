// Function: sub_80B840
// Address: 0x80b840
//
_BOOL8 __fastcall sub_80B840(__int64 a1, const char *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  const char *v6; // rdi
  __int64 *v7; // rbx
  _QWORD *v8; // r12
  __int64 v9; // rcx
  _QWORD *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // rax

  v2 = a1;
  v3 = sub_809820(a1);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(v3 + 40);
  if ( !v4 )
    return 0;
  if ( *(_BYTE *)(v4 + 28) != 3 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(v4 + 32) + 124LL) & 0x10) == 0 )
    return 0;
  v6 = *(const char **)(v3 + 8);
  if ( !v6 || strcmp(v6, a2) )
    return 0;
  while ( *(_BYTE *)(v2 + 140) == 12 )
    v2 = *(_QWORD *)(v2 + 160);
  if ( (v7 = *(__int64 **)(*(_QWORD *)(v2 + 168) + 168LL)) != 0
    && !*((_BYTE *)v7 + 8)
    && ((v8 = (_QWORD *)v7[4], v10 = sub_72BA30(0), v8 == v10) || (unsigned int)sub_8D97D0(v8, v10, 0, v9, v11))
    && (v12 = *v7) != 0
    && !*(_BYTE *)(v12 + 8)
    && !*(_QWORD *)v12 )
  {
    return (unsigned int)sub_809870(*(_QWORD *)(v12 + 32), "char_traits") != 0;
  }
  else
  {
    return 0;
  }
}
