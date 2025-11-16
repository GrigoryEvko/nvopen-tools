// Function: sub_D47930
// Address: 0xd47930
//
__int64 __fastcall sub_D47930(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rdx

  v1 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 16LL);
  if ( v1 )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(v1 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v2 - 30) <= 0xAu )
        break;
      v1 = *(_QWORD *)(v1 + 8);
      if ( !v1 )
        return 0;
    }
    v3 = 0;
    v4 = *(_QWORD *)(v2 + 40);
    if ( !*(_BYTE *)(a1 + 84) )
      goto LABEL_13;
    while ( 1 )
    {
      v5 = *(_QWORD **)(a1 + 64);
      v6 = &v5[*(unsigned int *)(a1 + 76)];
      if ( v5 != v6 )
      {
        while ( v4 != *v5 )
        {
          if ( v6 == ++v5 )
            goto LABEL_10;
        }
        if ( v3 )
          return 0;
LABEL_9:
        v3 = v4;
      }
      while ( 1 )
      {
        do
        {
LABEL_10:
          v1 = *(_QWORD *)(v1 + 8);
          if ( !v1 )
            return v3;
          v7 = *(_QWORD *)(v1 + 24);
        }
        while ( (unsigned __int8)(*(_BYTE *)v7 - 30) > 0xAu );
        v4 = *(_QWORD *)(v7 + 40);
        if ( *(_BYTE *)(a1 + 84) )
          break;
LABEL_13:
        if ( sub_C8CA60(a1 + 56, v4) )
        {
          if ( v3 )
            return 0;
          goto LABEL_9;
        }
      }
    }
  }
  return 0;
}
