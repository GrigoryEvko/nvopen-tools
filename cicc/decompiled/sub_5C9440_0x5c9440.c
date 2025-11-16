// Function: sub_5C9440
// Address: 0x5c9440
//
int __fastcall sub_5C9440(__int64 a1)
{
  const char *v2; // r14
  size_t v3; // rax
  const char *v4; // rdi
  size_t v5; // r12
  _UNKNOWN **v6; // rbx
  int result; // eax

  v2 = *(const char **)(a1 + 16);
  v3 = strlen(v2);
  v4 = (const char *)off_4A428A0;
  if ( off_4A428A0 )
  {
    v5 = v3;
    v6 = &off_4A428A0;
    while ( 1 )
    {
      result = strncmp(v4, v2, v5);
      if ( !result )
        break;
      v4 = (const char *)v6[1];
      ++v6;
      if ( !v4 )
        goto LABEL_6;
    }
  }
  else
  {
LABEL_6:
    *(_BYTE *)(a1 + 11) |= 0x80u;
    return sub_684B10(2803, a1 + 56, v2);
  }
  return result;
}
