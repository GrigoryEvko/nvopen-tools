// Function: sub_171BF50
// Address: 0x171bf50
//
__int64 __fastcall sub_171BF50(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v7; // rdi
  unsigned int v9; // r14d
  char *v10; // r12

  v7 = *(_QWORD *)a1;
  if ( v7 && (v9 = sub_171A810(v7, a2, a3, a4, a5, a6)) != 0 )
  {
    if ( *(_BYTE *)(a1 + 8) || *(_WORD *)(a1 + 10) != 1 )
    {
      v10 = (char *)(a1 + 8);
      sub_171BB20(a2 + 8, v10, a4, a5, a6);
      if ( v9 == 2 )
        sub_171BB20(a3 + 8, v10, a4, a5, a6);
    }
  }
  else
  {
    return 0;
  }
  return v9;
}
