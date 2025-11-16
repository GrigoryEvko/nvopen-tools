// Function: sub_889000
// Address: 0x889000
//
__int64 __fastcall sub_889000(unsigned __int8 a1, unsigned __int16 a2, int a3)
{
  char *v4; // rax
  char v5; // al
  unsigned int v6; // r12d
  int v8; // [rsp+0h] [rbp-20h] BYREF
  int v9; // [rsp+4h] [rbp-1Ch] BYREF
  char *v10; // [rsp+8h] [rbp-18h] BYREF

  v10 = 0;
  if ( a1 == 9 )
  {
    v8 = 0;
    v9 = 0;
    sub_888610(off_4A52088[4 * a2], &v8, &v9, &v10, 0);
    v4 = v10;
  }
  else
  {
    v4 = *(char **)(unk_4D03FB0 + 24LL * *(unsigned __int16 *)(qword_4A598E0[a1] + 16LL * a2 + 8));
    v10 = v4;
  }
  if ( !v4 )
    return 1;
  v5 = *v4;
  v6 = 1;
  if ( !v5 || v5 == 93 )
    return v6;
  do
  {
    if ( v5 == 105 )
    {
      if ( !unk_4D04290 )
      {
        v6 = 0;
        if ( a3 )
          sub_6851C0(0xAE5u, &dword_4F063F8);
      }
    }
    else if ( v5 > 105 )
    {
      if ( v5 != 118 )
LABEL_20:
        sub_721090();
    }
    else if ( v5 == 99 )
    {
      if ( !dword_4D041B4 )
      {
        v6 = 0;
        if ( a3 )
          sub_6851C0(0xC4Eu, &dword_4F063F8);
      }
    }
    else
    {
      if ( v5 != 102 )
        goto LABEL_20;
      if ( !dword_4D04284 )
      {
        v6 = 0;
        if ( a3 )
          sub_6851C0(0xB53u, &dword_4F063F8);
      }
    }
    v5 = *++v10;
  }
  while ( *v10 && v5 != 93 );
  if ( (a3 & (v6 ^ 1)) == 0 )
    return v6;
  word_4F06418[0] = 1;
  sub_885B10((__int64)&qword_4D04A00);
  return 0;
}
