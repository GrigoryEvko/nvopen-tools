// Function: sub_16E90A0
// Address: 0x16e90a0
//
char *__fastcall sub_16E90A0(__int64 a1, unsigned __int64 a2, int a3, int a4, int a5, int a6)
{
  char *result; // rax

  if ( a2 <= 0x1FFFFFFFFFFFFFFFLL && (result = realloc(*(_QWORD *)(a1 + 24), 8 * a2, a3, a4, a5, a6)) != 0 )
  {
    *(_QWORD *)(a1 + 32) = a2;
    *(_QWORD *)(a1 + 24) = result;
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 12;
    *(_QWORD *)a1 = &unk_4FA17D0;
    *(_QWORD *)(a1 + 8) = &unk_4FA17D0;
    return (char *)&unk_4FA17D0;
  }
  return result;
}
