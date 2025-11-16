// Function: sub_5D34A0
// Address: 0x5d34a0
//
int sub_5D34A0()
{
  unsigned __int64 v0; // rax
  int result; // eax
  int v2; // edi
  int v3; // r12d
  char *v4; // rbx
  char s[8]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v6; // [rsp+8h] [rbp-48h]
  __int128 v7; // [rsp+10h] [rbp-40h]
  __int128 v8; // [rsp+20h] [rbp-30h]
  __int16 v9; // [rsp+30h] [rbp-20h]

  *(_QWORD *)s = 5529439;
  v6 = 0;
  v9 = 0;
  v7 = 0;
  v8 = 0;
  v0 = sub_737880();
  if ( v0 > 9 )
  {
    sub_622470(v0, &s[3]);
  }
  else
  {
    s[4] = 0;
    s[3] = v0 + 48;
  }
  result = strlen(s);
  v2 = s[0];
  v3 = result;
  if ( s[0] )
  {
    v4 = &s[1];
    do
    {
      ++v4;
      result = putc(v2, stream);
      v2 = *(v4 - 1);
    }
    while ( *(v4 - 1) );
  }
  dword_4CF7F40 += v3;
  return result;
}
