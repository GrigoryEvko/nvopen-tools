// Function: sub_D5D290
// Address: 0xd5d290
//
__int64 __fastcall sub_D5D290(__int64 a1, int a2)
{
  char *v2; // rax
  int i; // edx
  __int64 v5; // [rsp+4h] [rbp-Ch]

  v2 = (char *)&unk_3F71EE0;
  for ( i = 30; ; i = *(_DWORD *)v2 )
  {
    if ( a2 == i )
      goto LABEL_11;
    if ( a2 == *((_DWORD *)v2 + 3) )
    {
      v2 += 12;
      goto LABEL_11;
    }
    if ( a2 == *((_DWORD *)v2 + 6) )
    {
      v2 += 24;
      goto LABEL_11;
    }
    if ( a2 == *((_DWORD *)v2 + 9) )
    {
      v2 += 36;
LABEL_11:
      if ( v2 == (char *)&unk_3F7203C )
        return v5;
      return *(_QWORD *)(v2 + 4);
    }
    v2 += 48;
    if ( v2 == (char *)&unk_3F72030 )
      break;
  }
  if ( a2 != 29 )
    return v5;
  return *(_QWORD *)(v2 + 4);
}
