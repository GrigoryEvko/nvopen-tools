// Function: sub_1CE34D0
// Address: 0x1ce34d0
//
__int64 __fastcall sub_1CE34D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  unsigned int v5; // edx
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v9; // r13
  __int64 v10; // r12
  _QWORD **v11; // rax
  unsigned __int8 v12; // [rsp+Fh] [rbp-31h]

  v4 = sub_1CDE330(a1, a2);
  v5 = 1;
  if ( v4 >= 0 )
  {
    v6 = *(__int64 **)(a3 + 8);
    v7 = &v6[*(unsigned int *)(a3 + 24)];
    if ( !*(_DWORD *)(a3 + 16) || v6 == v7 )
      return 1;
    while ( 1 )
    {
      LOBYTE(v5) = *v6 == -16 || *v6 == -8;
      if ( !(_BYTE)v5 )
        break;
      if ( ++v6 == v7 )
        return 1;
    }
    if ( v6 == v7 )
      return 1;
    v9 = 1LL << v4;
    v10 = 8LL * ((unsigned int)v4 >> 6);
LABEL_11:
    v12 = v5;
    v11 = (_QWORD **)sub_1BFDF20(a1, *v6);
    v5 = v12;
    if ( (*(_QWORD *)(**v11 + v10) & v9) != 0 )
    {
      while ( 1 )
      {
        if ( ++v6 == v7 )
          return 1;
        if ( *v6 != -16 && *v6 != -8 )
        {
          if ( v6 != v7 )
            goto LABEL_11;
          return 1;
        }
      }
    }
  }
  return v5;
}
