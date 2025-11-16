// Function: sub_140A0D0
// Address: 0x140a0d0
//
__int64 __fastcall sub_140A0D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 i; // r15
  unsigned __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int64 *v10; // rax
  __int64 v11; // rax
  unsigned __int64 *v12; // rsi
  unsigned int v13; // edi
  unsigned __int64 *v14; // rcx

  v2 = a2 + 72;
  v3 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = v3;
  if ( a2 + 72 == v4 )
  {
    i = 0;
    while ( 1 )
    {
LABEL_9:
      if ( v2 == v4 )
        return 0;
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 8) == 54 )
      {
        v8 = *(_QWORD *)(i - 48);
        if ( (unsigned __int8)sub_13F8680(v8, v5, 0, 0) )
        {
          v11 = *(unsigned int *)(a1 + 168);
          if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 172) )
          {
            sub_16CD150(a1 + 160, a1 + 176, 0, 8);
            v11 = *(unsigned int *)(a1 + 168);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v11) = v8;
          ++*(_DWORD *)(a1 + 168);
        }
        if ( (unsigned __int8)sub_13F8190(v8, 1 << (*(unsigned __int16 *)(i - 6) >> 1) >> 1, v5, 0, 0) )
        {
          v10 = *(unsigned __int64 **)(a1 + 216);
          if ( *(unsigned __int64 **)(a1 + 224) != v10 )
            goto LABEL_22;
          v12 = &v10[*(unsigned int *)(a1 + 236)];
          v13 = *(_DWORD *)(a1 + 236);
          if ( v10 == v12 )
          {
LABEL_34:
            if ( v13 < *(_DWORD *)(a1 + 232) )
            {
              *(_DWORD *)(a1 + 236) = v13 + 1;
              *v12 = v8;
              ++*(_QWORD *)(a1 + 208);
              goto LABEL_14;
            }
LABEL_22:
            sub_16CCBA0(a1 + 208, v8);
            goto LABEL_14;
          }
          v14 = 0;
          while ( v8 != *v10 )
          {
            if ( *v10 == -2 )
              v14 = v10;
            if ( v12 == ++v10 )
            {
              if ( !v14 )
                goto LABEL_34;
              *v14 = v8;
              --*(_DWORD *)(a1 + 240);
              ++*(_QWORD *)(a1 + 208);
              break;
            }
          }
        }
      }
LABEL_14:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v4 + 24) )
      {
        v9 = v4 - 24;
        if ( !v4 )
          v9 = 0;
        if ( i != v9 + 40 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v2 == v4 )
          return 0;
        if ( !v4 )
          BUG();
      }
    }
  }
  if ( !v4 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v4 + 24);
    if ( i != v4 + 16 )
      goto LABEL_9;
    v4 = *(_QWORD *)(v4 + 8);
    if ( v2 == v4 )
      return 0;
    if ( !v4 )
      BUG();
  }
}
