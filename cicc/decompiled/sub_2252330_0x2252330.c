// Function: sub_2252330
// Address: 0x2252330
//
__int64 __fastcall sub_2252330(__int64 a1, _QWORD *a2, __int64 a3, unsigned int a4)
{
  const char *v6; // rdi
  const char *v7; // rsi

  v6 = *(const char **)(a1 + 8);
  v7 = (const char *)a2[1];
  if ( v6 == v7 || *v6 != 42 && !strcmp(v6, v7) )
    return 1;
  if ( a4 > 3 )
    return 0;
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 40LL))(a2, a1, a3);
}
