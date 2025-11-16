// Function: sub_2052F00
// Address: 0x2052f00
//
char *__fastcall sub_2052F00(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  int v6; // eax

  if ( !*(_QWORD *)(*(_QWORD *)(a1 + 712) + 32LL) )
    return sub_1DD8D40(a2, a3);
  if ( a4 != -1 )
    return sub_1DD8FE0(a2, a3, a4);
  v6 = sub_2052E90(a1, a2, a3);
  return sub_1DD8FE0(a2, a3, v6);
}
