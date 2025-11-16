// Function: sub_131D480
// Address: 0x131d480
//
__int64 __fastcall sub_131D480(__int64 a1, char **a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  char *v7; // r8
  char *i; // rax
  unsigned __int64 v9; // rdx

  if ( a4 )
  {
    v6 = 0;
    v7 = (char *)&unk_497FBA0;
    for ( i = (char *)&unk_497FB60; ; v7 = (char *)*((_QWORD *)i + 3) )
    {
      if ( *v7 )
      {
        v9 = *(_QWORD *)(a3 + 8 * v6);
        if ( *((_QWORD *)i + 2) <= v9 )
          return 2;
        i = &v7[40 * v9];
      }
      else
      {
        i = (char *)(*((__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))v7 + 1))(
                      a1,
                      a3,
                      a4,
                      *(_QWORD *)(a3 + 8 * v6));
        if ( !i )
          return 2;
      }
      if ( a4 == ++v6 )
        goto LABEL_11;
    }
  }
  i = (char *)&unk_497FB60;
LABEL_11:
  *a2 = i;
  return 0;
}
