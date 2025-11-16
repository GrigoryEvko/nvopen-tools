// Function: sub_E22480
// Address: 0xe22480
//
size_t __fastcall sub_E22480(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // rax
  _BYTE *v4; // r12
  size_t v5; // rbx

  v3 = *a2;
  if ( *a2 )
  {
    v4 = (_BYTE *)a2[1];
    v5 = 0;
    while ( v4[v5] != 64 )
    {
      if ( ++v5 == v3 )
        goto LABEL_5;
    }
    if ( !v5 )
    {
LABEL_5:
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
    a2[1] = (__int64)&v4[v5 + 1];
    *a2 = v3 - (v5 + 1);
    if ( a3 )
      sub_E21AF0(a1, v5, v4);
    return v5;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
}
