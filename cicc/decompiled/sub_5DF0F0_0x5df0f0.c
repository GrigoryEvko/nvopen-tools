// Function: sub_5DF0F0
// Address: 0x5df0f0
//
void __fastcall sub_5DF0F0(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 i; // rax
  int v7; // edi
  char *v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9

  if ( !(unsigned int)sub_8D2E30(*a1) )
    goto LABEL_5;
  for ( i = sub_8D46C0(*a1); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(i + 142) & 0x10) != 0 )
  {
    v7 = 40;
    v8 = "(char *)";
    do
    {
      ++v8;
      putc(v7, stream);
      v7 = *(v8 - 1);
    }
    while ( *(v8 - 1) );
    dword_4CF7F40 += 9;
    sub_5DBFC0((__int64)a1, (FILE *)1, v9, v10, v11, v12);
    putc(41, stream);
    ++dword_4CF7F40;
  }
  else
  {
LABEL_5:
    sub_5DBFC0((__int64)a1, (FILE *)1, v2, v3, v4, v5);
  }
}
