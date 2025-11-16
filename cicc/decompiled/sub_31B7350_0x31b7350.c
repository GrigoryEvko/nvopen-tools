// Function: sub_31B7350
// Address: 0x31b7350
//
__int64 *__fastcall sub_31B7350(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rax

  v1 = a1[2];
  v2 = a1[3] + 1;
  a1[3] = v2;
  v3 = *(unsigned int *)(v1 + 8);
  if ( v2 >= v3 )
  {
    v4 = a1[1];
    v5 = *a1;
    a1[3] = 0;
    a1[1] = v4 + 88;
    if ( v4 + 88 == *(_QWORD *)(v5 + 32) + 88LL * *(unsigned int *)(v5 + 40) )
    {
      a1[2] = 0;
      return a1;
    }
    v1 = v4 + 112;
    v2 = 0;
    a1[2] = v4 + 112;
    v3 = *(unsigned int *)(v4 + 120);
  }
  while ( v3 > v2
       && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2) + 144LL) == *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v1 + 8 * v2)
                                                                                + 16LL) )
  {
    v6 = v2 + 1;
    a1[3] = v6;
    if ( v6 >= *(unsigned int *)(v1 + 8) )
    {
      v7 = a1[1];
      v8 = *a1;
      a1[3] = 0;
      v9 = v7 + 88;
      v10 = v7 + 112;
      a1[1] = v9;
      if ( v9 == *(_QWORD *)(v8 + 32) + 88LL * *(unsigned int *)(v8 + 40) )
        v10 = 0;
      a1[2] = v10;
    }
    sub_31B7070(a1);
    v1 = a1[2];
    if ( !v1 )
      break;
    v2 = a1[3];
    v3 = *(unsigned int *)(v1 + 8);
  }
  return a1;
}
