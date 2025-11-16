// Function: sub_7F7BA0
// Address: 0x7f7ba0
//
__int64 __fastcall sub_7F7BA0(__int64 a1)
{
  __int64 v1; // rbx
  char v2; // al
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r14
  const char *v7; // rsi
  __int64 v8; // r15
  __int64 i; // r13
  __int64 j; // r15
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // [rsp-40h] [rbp-40h] BYREF

  if ( !a1 )
    return 1;
  v1 = a1;
  do
  {
    v2 = *(_BYTE *)(v1 + 40);
    if ( v2 == 8 )
    {
      if ( *(_QWORD *)(v1 + 48) )
        return 0;
    }
    else if ( v2 )
    {
      if ( v2 != 11 || !(unsigned int)sub_7F7BA0(*(_QWORD *)(v1 + 72)) )
        return 0;
    }
    else
    {
      v3 = *(_QWORD *)(v1 + 48);
      if ( *(_BYTE *)(v3 + 24) == 1 && *(_BYTE *)(v3 + 56) == 105 )
      {
        v13 = *(_QWORD *)(v3 + 72);
        if ( *(_BYTE *)(v13 + 24) != 20 )
          return 0;
        v14 = *(_QWORD *)(v13 + 56);
        if ( (unsigned __int8)(*(_BYTE *)(v14 + 174) - 1) > 1u )
          return 0;
        v15 = *(_QWORD *)(v13 + 16);
        if ( !v15
          || *(_QWORD *)(v15 + 16)
          || !(unsigned int)sub_7F5130(v15)
          || sub_7E5340(v14)
          || !(unsigned int)sub_7F7E80(v14) )
        {
          return 0;
        }
      }
      else
      {
        v19 = 0;
        v4 = sub_6E8430(v3);
        if ( *(_BYTE *)(v4 + 56) != 91 )
          return 0;
        v5 = *(_QWORD *)(v4 + 72);
        if ( *(_BYTE *)(v5 + 24) != 1 )
          return 0;
        if ( *(_BYTE *)(v5 + 56) != 105 )
          return 0;
        v7 = *(const char **)(sub_72B0F0(*(_QWORD *)(v5 + 72), 0) + 8);
        if ( !v7 || strcmp(v7, "__cxa_vec_ctor") )
          return 0;
        v8 = *(_QWORD *)(*(_QWORD *)(v5 + 72) + 16LL);
        for ( i = v8; sub_730740(i); i = *(_QWORD *)(i + 72) )
          ;
        if ( (unsigned int)sub_7E2550(i, &v19) )
        {
          for ( i = *(_QWORD *)(*(_QWORD *)(i + 72) + 16LL); sub_730740(i); i = *(_QWORD *)(i + 72) )
            ;
        }
        if ( *(_BYTE *)(i + 24) != 1 )
          return 0;
        if ( *(_BYTE *)(i + 56) != 21 )
          return 0;
        if ( !(unsigned int)sub_7F5130(*(_QWORD *)(i + 72)) )
          return 0;
        for ( j = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 16) + 16LL) + 16LL); sub_730740(j); j = *(_QWORD *)(j + 72) )
          ;
        if ( *(_BYTE *)(j + 24) != 20 )
          return 0;
        v11 = *(_QWORD *)(j + 56);
        if ( *(_BYTE *)(v11 + 174) != 1 || sub_7E5340(v11) || !(unsigned int)sub_7F7E80(v11) )
          return 0;
        v12 = *(__int64 **)(v5 + 16);
        if ( v19 )
        {
          if ( *((_BYTE *)v12 + 24) != 3 || v19 != v12[7] )
            return 0;
        }
        else
        {
          v16 = *(_QWORD *)i;
          v17 = *v12;
          if ( *(_QWORD *)i != *v12 )
          {
            if ( !v16 )
              return 0;
            if ( !v17 )
              return 0;
            if ( !dword_4F07588 )
              return 0;
            v18 = *(_QWORD *)(v16 + 32);
            if ( *(_QWORD *)(v17 + 32) != v18 || !v18 )
              return 0;
          }
          if ( *((_BYTE *)v12 + 24) != 1 || *((_BYTE *)v12 + 56) != 21 || !(unsigned int)sub_7F5130(v12[9]) )
            return 0;
        }
      }
    }
    v1 = *(_QWORD *)(v1 + 16);
  }
  while ( v1 );
  return 1;
}
