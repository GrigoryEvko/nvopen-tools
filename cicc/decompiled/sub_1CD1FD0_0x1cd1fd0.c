// Function: sub_1CD1FD0
// Address: 0x1cd1fd0
//
__int64 __fastcall sub_1CD1FD0(__int64 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r13
  _QWORD *v4; // rbx
  __int64 v5; // rcx
  unsigned int v6; // edx
  _QWORD *v7; // rax
  unsigned int v8; // r12d
  __int64 v9; // r15
  __int64 v10; // r14
  unsigned int v11; // [rsp+Ch] [rbp-34h]

  if ( !*(_DWORD *)(a1 + 96) )
    return 0;
  v2 = *(_QWORD **)(a1 + 88);
  v3 = &v2[2 * *(unsigned int *)(a1 + 104)];
  if ( v2 == v3 )
    return 0;
  while ( 1 )
  {
    v4 = v2;
    if ( *v2 != -16 && *v2 != -8 )
      break;
    v2 += 2;
    if ( v3 == v2 )
      return 0;
  }
  if ( v3 == v2 )
  {
    return 0;
  }
  else
  {
    v11 = 0;
    do
    {
      v5 = *(_QWORD *)v4[1];
      v6 = (unsigned int)(*(_DWORD *)(v5 + 16) + 63) >> 6;
      if ( v6 )
      {
        v7 = *(_QWORD **)v5;
        v8 = 0;
        v9 = *(_QWORD *)v5 + 8LL;
        v10 = v9 + 8LL * (v6 - 1);
        while ( 1 )
        {
          v8 += sub_39FAC40(*v7);
          v7 = (_QWORD *)v9;
          if ( v10 == v9 )
            break;
          v9 += 8;
        }
        if ( v11 >= v8 )
          v8 = v11;
        v11 = v8;
      }
      v4 += 2;
      if ( v4 == v3 )
        break;
      while ( *v4 == -16 || *v4 == -8 )
      {
        v4 += 2;
        if ( v3 == v4 )
          return v11;
      }
    }
    while ( v3 != v4 );
  }
  return v11;
}
