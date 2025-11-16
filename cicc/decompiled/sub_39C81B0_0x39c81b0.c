// Function: sub_39C81B0
// Address: 0x39c81b0
//
__int64 __fastcall sub_39C81B0(void *s1, size_t n, __int64 *a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r14
  unsigned int i; // r13d
  const char *v8; // rax
  __int64 v9; // rdx

  if ( !a3 )
    return (unsigned int)-1;
  v3 = *a3;
  if ( !(unsigned __int8)sub_1C2F070(*a3) )
    return (unsigned int)-1;
  if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
  {
    sub_15E08E0(v3, n);
    v4 = *(_QWORD *)(v3 + 88);
    if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
      sub_15E08E0(v3, n);
    v5 = *(_QWORD *)(v3 + 88);
  }
  else
  {
    v4 = *(_QWORD *)(v3 + 88);
    v5 = v4;
  }
  v6 = v5 + 40LL * *(_QWORD *)(v3 + 96);
  if ( v6 != v4 )
  {
    for ( i = 0; ; ++i )
    {
      v8 = sub_1649960(v4);
      if ( v9 == n && (!n || !memcmp(s1, v8, n)) )
        break;
      v4 += 40;
      if ( v4 == v6 )
        return (unsigned int)-1;
    }
  }
  else
  {
    return (unsigned int)-1;
  }
  return i;
}
