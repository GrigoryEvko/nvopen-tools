// Function: sub_771080
// Address: 0x771080
//
__int64 __fastcall sub_771080(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 result; // rax
  unsigned __int64 *v4; // r9
  unsigned __int64 *v5; // r10
  __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int j; // edx
  __int64 v13; // rax
  unsigned int i; // edi
  __int64 v15; // rsi

  result = 0;
  if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
  {
    v4 = *(unsigned __int64 **)a2;
    v5 = *(unsigned __int64 **)(a2 + 24);
    result = 1;
    if ( *(unsigned __int64 **)a2 != v5
      && ((unsigned __int8)(1 << (((_BYTE)v4 - (_BYTE)v5) & 7))
        & *((_BYTE *)v5 - ((unsigned int)((_DWORD)v4 - (_DWORD)v5) >> 3) - 10)) != 0 )
    {
      v8 = *a1;
      do
      {
        v9 = *v4;
        if ( !*v4 )
          break;
        v10 = *(_QWORD **)(v9 + 120);
        v11 = v9 >> 3;
        if ( v10 )
        {
          while ( v10[2] != v8 )
          {
            v10 = (_QWORD *)*v10;
            if ( !v10 )
              goto LABEL_12;
          }
          if ( v10[4] )
          {
            for ( i = v11 & qword_4F08388; ; i = qword_4F08388 & (i + 1) )
            {
              v15 = qword_4F08380 + 16LL * i;
              if ( v9 == *(_QWORD *)v15 )
              {
                *a3 += *(_DWORD *)(v15 + 8);
                goto LABEL_11;
              }
              if ( !*(_QWORD *)v15 )
                break;
            }
            *a3 = *a3;
          }
LABEL_11:
          v8 = v10[1];
        }
LABEL_12:
        for ( j = qword_4F08388 & v11; ; j = qword_4F08388 & (j + 1) )
        {
          v13 = qword_4F08380 + 16LL * j;
          if ( v9 == *(_QWORD *)v13 )
            break;
          if ( !*(_QWORD *)v13 )
            goto LABEL_17;
        }
        v4 = (unsigned __int64 *)((char *)v4 - *(unsigned int *)(v13 + 8));
LABEL_17:
        ;
      }
      while ( ((unsigned __int8)(1 << (((_BYTE)v4 - (_BYTE)v5) & 7))
             & *((_BYTE *)v5 - ((unsigned int)((_DWORD)v4 - (_DWORD)v5) >> 3) - 10)) != 0
           && v5 != v4 );
      *a1 = v8;
      result = 1;
      *(_QWORD *)a2 = v4;
    }
  }
  return result;
}
