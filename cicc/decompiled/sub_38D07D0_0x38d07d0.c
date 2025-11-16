// Function: sub_38D07D0
// Address: 0x38d07d0
//
_QWORD *__fastcall sub_38D07D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  char *v3; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax

  if ( *(_DWORD *)(a2 + 44) )
  {
    if ( a3 < 0 )
    {
      a3 = -a3;
      v6 = a3;
      if ( (unsigned __int64)a3 >> 60 )
        goto LABEL_15;
      do
        v6 *= 16LL;
      while ( !(v6 >> 60) );
      if ( v6 >> 60 <= 9 )
LABEL_15:
        v3 = "-%lxh";
      else
        v3 = "-0%lxh";
    }
    else
    {
      v5 = a3;
      if ( !a3 || (unsigned __int64)a3 >> 60 )
        goto LABEL_16;
      do
        v5 *= 16LL;
      while ( !(v5 >> 60) );
      if ( v5 >> 60 <= 9 )
LABEL_16:
        v3 = "%lxh";
      else
        v3 = "0%lxh";
    }
    goto LABEL_3;
  }
  v3 = "0x%lx";
  if ( a3 >= 0 )
  {
LABEL_3:
    a1[1] = v3;
    a1[2] = a3;
    *a1 = &unk_49EEAD0;
    return a1;
  }
  a1[1] = "-0x%lx";
  a1[2] = -a3;
  *a1 = &unk_49EEAD0;
  return a1;
}
