// Function: sub_5D43B0
// Address: 0x5d43b0
//
int __fastcall sub_5D43B0(unsigned int a1, __int64 a2)
{
  char *v2; // r15
  int v3; // eax
  int v4; // edi
  int v5; // r12d
  _BOOL4 v6; // r12d
  int result; // eax
  char s[8]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+8h] [rbp-98h]
  __int128 v10; // [rsp+10h] [rbp-90h]
  __int128 v11; // [rsp+20h] [rbp-80h]
  __int128 v12; // [rsp+30h] [rbp-70h]
  __int128 v13; // [rsp+40h] [rbp-60h]
  __int128 v14; // [rsp+50h] [rbp-50h]
  int v15; // [rsp+60h] [rbp-40h]

  v11 = 0;
  *(_QWORD *)s = 0x20656E696C23LL;
  v9 = 0;
  v15 = 0;
  v10 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  if ( dword_4CF7F40 )
    sub_5D37C0();
  dword_4CF7F44 = a1;
  dword_4CF7F3C = 1;
  if ( unk_4F068C4 | unk_4D04934 )
  {
    s[1] = 32;
    if ( a1 > 9 )
    {
      sub_622470(a1, &s[2]);
    }
    else
    {
      s[3] = 0;
      s[2] = a1 + 48;
    }
  }
  else if ( a1 > 9 )
  {
    sub_622470(a1, &s[6]);
  }
  else
  {
    s[7] = 0;
    s[6] = a1 + 48;
  }
  v2 = &s[1];
  v3 = strlen(s);
  v4 = s[0];
  v5 = v3;
  if ( s[0] )
  {
    do
    {
      ++v2;
      putc(v4, stream);
      v4 = *(v2 - 1);
    }
    while ( *(v2 - 1) );
  }
  qword_4CF7F48 = a2;
  dword_4CF7F40 += v5;
  v6 = unk_4D04934 == 0;
  putc(32, stream);
  putc(34, stream);
  sub_723850(*(_QWORD *)qword_4CF7F48, stream, v6, 1);
  putc(34, stream);
  if ( unk_4F068C4 && (*(_BYTE *)(a2 + 72) & 0x40) != 0 )
  {
    putc(32, stream);
    putc(51, stream);
  }
  result = putc(10, stream);
  dword_4CF7F40 = 0;
  return result;
}
