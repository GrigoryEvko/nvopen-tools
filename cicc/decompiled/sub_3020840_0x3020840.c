// Function: sub_3020840
// Address: 0x3020840
//
__int64 __fastcall sub_3020840(__int64 a1, __int64 a2, size_t a3)
{
  __int64 *v3; // rax
  const void *v4; // r14
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rbx
  const char *v10; // rax

  v3 = *(__int64 **)(a1 + 232);
  if ( !v3 )
    return 0xFFFFFFFFLL;
  v4 = (const void *)a2;
  v6 = *v3;
  if ( !(unsigned __int8)sub_CE9220(*v3) || !*(_QWORD *)(v6 + 104) )
    return 0xFFFFFFFFLL;
  v9 = 0;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v6 + 2) & 1) != 0 )
      sub_B2C6D0(v6, a2, v7, v8);
    v10 = sub_BD5D20(*(_QWORD *)(v6 + 96) + 40LL * (unsigned int)v9);
    if ( a3 == v7 )
    {
      if ( !a3 )
        break;
      a2 = (__int64)v10;
      if ( !memcmp(v4, v10, a3) )
        break;
    }
    if ( *(_QWORD *)(v6 + 104) <= (unsigned __int64)++v9 )
      return 0xFFFFFFFFLL;
  }
  return (unsigned int)v9;
}
