// Function: sub_1840ED0
// Address: 0x1840ed0
//
void __fastcall sub_1840ED0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v3; // eax
  __int64 v5; // rbx
  __int64 i; // r13
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // rdi
  unsigned int v10; // r8d
  __int64 *v11; // rcx

  if ( a1 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    v3 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
    if ( v3 )
    {
      v5 = v3 - 1;
      for ( i = 0; ; ++i )
      {
        v7 = sub_1649F00(*(_QWORD *)(v2 + 24 * (i - v3)));
        v8 = *(__int64 **)(a2 + 8);
        if ( *(__int64 **)(a2 + 16) != v8 )
          break;
        v9 = &v8[*(unsigned int *)(a2 + 28)];
        v10 = *(_DWORD *)(a2 + 28);
        if ( v8 == v9 )
        {
LABEL_17:
          if ( v10 >= *(_DWORD *)(a2 + 24) )
            break;
          *(_DWORD *)(a2 + 28) = v10 + 1;
          *v9 = v7;
          ++*(_QWORD *)a2;
        }
        else
        {
          v11 = 0;
          while ( v7 != *v8 )
          {
            if ( *v8 == -2 )
              v11 = v8;
            if ( v9 == ++v8 )
            {
              if ( !v11 )
                goto LABEL_17;
              *v11 = v7;
              --*(_DWORD *)(a2 + 32);
              ++*(_QWORD *)a2;
              if ( v5 != i )
                goto LABEL_6;
              return;
            }
          }
        }
LABEL_5:
        if ( v5 == i )
          return;
LABEL_6:
        v3 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
      }
      sub_16CCBA0(a2, v7);
      goto LABEL_5;
    }
  }
}
