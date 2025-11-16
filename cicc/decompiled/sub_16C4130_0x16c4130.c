// Function: sub_16C4130
// Address: 0x16c4130
//
unsigned __int64 __fastcall sub_16C4130(char *a1, unsigned __int64 a2, int a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r15
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  bool v9; // [rsp+Fh] [rbp-31h]

  v4 = sub_16C3BE0(a1, a2, a3);
  v9 = 1;
  v5 = v4;
  if ( a2 )
    v9 = !sub_16C36C0(a1[v4], a3);
  v6 = sub_16C3AD0(a1, a2, a3);
  while ( v5 && (v6 == -1 || v5 > v6) && sub_16C36C0(a1[v5 - 1], a3) )
    --v5;
  if ( v5 == v6 )
  {
    v7 = v6 + 1;
    if ( v9 )
      return v7;
  }
  return v5;
}
