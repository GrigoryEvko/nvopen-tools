// Function: sub_74A1C0
// Address: 0x74a1c0
//
__int64 __fastcall sub_74A1C0(__int64 *a1, unsigned int a2, void (__fastcall **a3)(char *))
{
  __int64 result; // rax
  __int64 *v5; // rbx

  result = a2;
  if ( a1 )
  {
    v5 = a1;
    do
    {
      while ( *((_BYTE *)v5 + 9) != 4 )
      {
        v5 = (__int64 *)*v5;
        if ( !v5 )
          return result;
      }
      if ( (_DWORD)result )
        goto LABEL_10;
      while ( 1 )
      {
        ((void (__fastcall *)(const char *, void (__fastcall **)(char *)))*a3)("_Alignas", a3);
        sub_74A070((__int64 *)v5[4], a3);
        v5 = (__int64 *)*v5;
        if ( !v5 )
          return 1;
        if ( *((_BYTE *)v5 + 9) != 4 )
          break;
LABEL_10:
        ((void (__fastcall *)(char *, void (__fastcall **)(char *)))*a3)(" ", a3);
      }
      v5 = (__int64 *)*v5;
      result = 1;
    }
    while ( v5 );
  }
  return result;
}
