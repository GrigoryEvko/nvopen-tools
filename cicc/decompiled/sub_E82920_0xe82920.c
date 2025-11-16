// Function: sub_E82920
// Address: 0xe82920
//
_QWORD *__fastcall sub_E82920(_QWORD *a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  char *v4; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  const char *v8; // rax

  v3 = *(_DWORD *)(a2 + 52);
  if ( v3 )
  {
    if ( v3 != 1 )
      BUG();
    if ( a3 >= 0 )
    {
      v6 = a3;
      if ( !a3 || (unsigned __int64)a3 >> 60 )
        goto LABEL_22;
      do
        v6 *= 16LL;
      while ( !(v6 >> 60) );
      if ( v6 >> 60 <= 9 )
LABEL_22:
        v4 = "%lxh";
      else
        v4 = "0%lxh";
      goto LABEL_3;
    }
    if ( a3 == 0x8000000000000000LL )
    {
      v4 = "-8000000000000000h";
      goto LABEL_3;
    }
    v7 = -a3;
    if ( !((unsigned __int64)-a3 >> 60) )
    {
      do
        v7 *= 16LL;
      while ( !(v7 >> 60) );
      if ( v7 >> 60 > 9 )
      {
        a1[2] = -a3;
        a1[1] = "-0%lxh";
        *a1 = &unk_49DBEF0;
        return a1;
      }
    }
    v8 = "-%lxh";
  }
  else
  {
    v4 = "0x%lx";
    if ( a3 >= 0 )
    {
LABEL_3:
      a1[1] = v4;
      a1[2] = a3;
      *a1 = &unk_49DBEF0;
      return a1;
    }
    if ( a3 == 0x8000000000000000LL )
    {
      v4 = "-0x8000000000000000";
      goto LABEL_3;
    }
    v8 = "-0x%lx";
  }
  a1[1] = v8;
  a1[2] = -a3;
  *a1 = &unk_49DBEF0;
  return a1;
}
