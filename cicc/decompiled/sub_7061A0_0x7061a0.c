// Function: sub_7061A0
// Address: 0x7061a0
//
_DWORD *sub_7061A0()
{
  _DWORD *result; // rax
  _DWORD *v1; // rax
  size_t v2; // rax
  char *v3; // rax

  sub_705870(0);
  if ( !unk_4D03FE8 )
  {
    result = (_DWORD *)dword_4D03C90;
    if ( !dword_4D03C90 )
      return result;
LABEL_5:
    sub_822170(dest);
    dword_4D03CB0[0] = 1;
    return dword_4D03CB0;
  }
  v1 = (_DWORD *)sub_7247C0(4);
  *v1 = 3550774;
  unk_4F07298 = v1;
  v2 = strlen(dest);
  v3 = (char *)sub_7247C0(v2 + 1);
  unk_4F072A0 = strcpy(v3, dest);
  unk_4F07318 = unk_4F06BA8;
  result = (_DWORD *)dword_4D03C90;
  if ( dword_4D03C90 )
    goto LABEL_5;
  return result;
}
