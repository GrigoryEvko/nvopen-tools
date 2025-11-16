// Function: sub_197D540
// Address: 0x197d540
//
__int64 __fastcall sub_197D540(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 *v8; // rdi
  unsigned int v9; // r8d
  __int64 *v10; // rcx

  if ( a1 != a2 )
  {
    for ( i = a1; a2 != i; i += 8 )
    {
LABEL_5:
      if ( *(_BYTE *)(*(_QWORD *)i + 16LL) == 55 )
      {
        v6 = *(_QWORD *)(*(_QWORD *)i - 24LL);
        v7 = *(__int64 **)(a3 + 8);
        if ( *(__int64 **)(a3 + 16) != v7 )
          goto LABEL_3;
        v8 = &v7[*(unsigned int *)(a3 + 28)];
        v9 = *(_DWORD *)(a3 + 28);
        if ( v7 != v8 )
        {
          v10 = 0;
          while ( v6 != *v7 )
          {
            if ( *v7 == -2 )
              v10 = v7;
            if ( v8 == ++v7 )
            {
              if ( !v10 )
                goto LABEL_16;
              i += 8;
              *v10 = v6;
              --*(_DWORD *)(a3 + 32);
              ++*(_QWORD *)a3;
              if ( a2 != i )
                goto LABEL_5;
              return a3;
            }
          }
          continue;
        }
LABEL_16:
        if ( v9 < *(_DWORD *)(a3 + 24) )
        {
          *(_DWORD *)(a3 + 28) = v9 + 1;
          *v8 = v6;
          ++*(_QWORD *)a3;
        }
        else
        {
LABEL_3:
          sub_16CCBA0(a3, v6);
        }
      }
    }
  }
  return a3;
}
