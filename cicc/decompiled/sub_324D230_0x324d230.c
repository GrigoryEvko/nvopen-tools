// Function: sub_324D230
// Address: 0x324d230
//
void __fastcall sub_324D230(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  char **v4; // rbx
  __int64 v5; // rax
  char **v6; // r14
  char *v7; // rdx
  char v8; // al

  if ( a3 )
  {
    v3 = *(_BYTE *)(a3 - 16);
    if ( (v3 & 2) != 0 )
    {
      v4 = *(char ***)(a3 - 32);
      v5 = *(unsigned int *)(a3 - 24);
    }
    else
    {
      v4 = (char **)(a3 - 16 - 8LL * ((v3 >> 2) & 0xF));
      v5 = (*(_WORD *)(a3 - 16) >> 6) & 0xF;
    }
    v6 = &v4[v5];
    while ( v6 != v4 )
    {
      while ( 1 )
      {
        v7 = *v4;
        v8 = **v4;
        if ( v8 != 23 )
          break;
        sub_324CDB0(a1, a2, (__int64)v7);
LABEL_7:
        if ( v6 == ++v4 )
          return;
      }
      if ( v8 != 24 )
        goto LABEL_7;
      ++v4;
      sub_324CF10(a1, a2, (__int64)v7);
    }
  }
}
