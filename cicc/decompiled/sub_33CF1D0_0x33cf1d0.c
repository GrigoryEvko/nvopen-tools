// Function: sub_33CF1D0
// Address: 0x33cf1d0
//
char __fastcall sub_33CF1D0(__int64 a1, int a2, __int64 a3)
{
  char result; // al
  __int64 v4; // rdi
  __int64 *v5; // rbx
  __int64 v6; // rax
  __int64 *v7; // r12
  signed __int64 v8; // rax
  __int64 *v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi

  result = 0;
  if ( (unsigned int)(a2 - 59) <= 3 )
  {
    v4 = *(_QWORD *)(a3 + 16);
    if ( *(_DWORD *)(v4 + 24) == 51 || sub_33CF170(v4) )
      return 1;
    result = sub_33CA6D0(v4);
    if ( result )
    {
      v5 = *(__int64 **)(v4 + 40);
      v6 = 5LL * *(unsigned int *)(v4 + 64);
      v7 = &v5[v6];
      v8 = 0xCCCCCCCCCCCCCCCDLL * ((v6 * 8) >> 3);
      if ( v8 >> 2 )
      {
        v9 = &v5[20 * (v8 >> 2)];
        while ( *(_DWORD *)(*v5 + 24) != 51 && !sub_33CF170(*v5) )
        {
          v10 = v5[5];
          if ( *(_DWORD *)(v10 + 24) == 51 || sub_33CF170(v10) )
            return v7 != v5 + 5;
          v11 = v5[10];
          if ( *(_DWORD *)(v11 + 24) == 51 || sub_33CF170(v11) )
            return v7 != v5 + 10;
          v12 = v5[15];
          if ( *(_DWORD *)(v12 + 24) == 51 || sub_33CF170(v12) )
            return v7 != v5 + 15;
          v5 += 20;
          if ( v9 == v5 )
          {
            v8 = 0xCCCCCCCCCCCCCCCDLL * (v7 - v5);
            goto LABEL_24;
          }
        }
        return v7 != v5;
      }
LABEL_24:
      if ( v8 != 2 )
      {
        if ( v8 != 3 )
        {
          if ( v8 != 1 )
            return 0;
          goto LABEL_34;
        }
        if ( *(_DWORD *)(*v5 + 24) == 51 || sub_33CF170(*v5) )
          return v7 != v5;
        v5 += 5;
      }
      if ( *(_DWORD *)(*v5 + 24) == 51 || sub_33CF170(*v5) )
        return v7 != v5;
      v5 += 5;
LABEL_34:
      if ( *(_DWORD *)(*v5 + 24) != 51 && !sub_33CF170(*v5) )
        return 0;
      return v7 != v5;
    }
  }
  return result;
}
