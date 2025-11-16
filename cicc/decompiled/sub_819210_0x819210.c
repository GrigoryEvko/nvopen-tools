// Function: sub_819210
// Address: 0x819210
//
__int64 __fastcall sub_819210(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  char v4; // r15
  size_t v5; // r13
  const char *v6; // r12
  char *s1; // [rsp+8h] [rbp-38h]

  *a2 = 0;
  if ( !a1 )
    return 0;
  v2 = a1;
  v3 = 0;
  s1 = (char *)qword_4F06410;
  v4 = *qword_4F06410;
  v5 = unk_4F06400;
  while ( 1 )
  {
    v6 = *(const char **)v2;
    ++v3;
    if ( v4 == **(_BYTE **)v2 && strlen(*(const char **)v2) == v5 && !strncmp(s1, v6, v5) )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  *a2 = v2;
  return v3;
}
