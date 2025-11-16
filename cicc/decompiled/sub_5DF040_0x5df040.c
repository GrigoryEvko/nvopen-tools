// Function: sub_5DF040
// Address: 0x5df040
//
void __fastcall sub_5DF040(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  if ( a2 )
  {
    putc(40, stream);
    ++dword_4CF7F40;
    if ( *(_BYTE *)(a1 + 24) == 1 )
      sub_5DBFC0(a1, (FILE *)(*(_BYTE *)(a1 + 56) == 73), v6, v7, v8, v9);
    else
      sub_5DBFC0(a1, 0, v6, v7, v8, v9);
    putc(41, stream);
    ++dword_4CF7F40;
  }
  else if ( *(_BYTE *)(a1 + 24) == 1 )
  {
    sub_5DBFC0(a1, (FILE *)(*(_BYTE *)(a1 + 56) == 73), a3, a4, a5, a6);
  }
  else
  {
    sub_5DBFC0(a1, 0, a3, a4, a5, a6);
  }
}
