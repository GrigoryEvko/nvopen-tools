// Function: sub_34B90F0
// Address: 0x34b90f0
//
__int64 __fastcall sub_34B90F0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  unsigned __int8 *v3; // r12
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx

  v2 = sub_BD3990(a1, a2);
  v3 = v2;
  if ( *v2 == 3 )
  {
    v5 = sub_BD5D20((__int64)v2);
    if ( v6 == 23 )
    {
      v7 = *(_QWORD *)v5 ^ 0x2E68652E6D766C6CLL;
      if ( !(v7 | *((_QWORD *)v5 + 1) ^ 0x6C612E6863746163LL)
        && *((_DWORD *)v5 + 4) == 1635135084
        && *((_WORD *)v5 + 10) == 30060
        && v5[22] == 101 )
      {
        v3 = (unsigned __int8 *)*((_QWORD *)v3 - 4);
        if ( *v3 >= 4u )
          return v7 | *((_QWORD *)v5 + 1) ^ 0x6C612E6863746163LL;
      }
    }
  }
  else if ( *v2 >= 3u )
  {
    return 0;
  }
  return (__int64)v3;
}
