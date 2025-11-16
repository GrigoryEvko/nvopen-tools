// Function: sub_1AC99A0
// Address: 0x1ac99a0
//
void __fastcall sub_1AC99A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 *v8; // rdi
  unsigned int v9; // r8d
  __int64 *v10; // rcx

  if ( a2 != a3 )
  {
    v5 = a2;
    do
    {
      v6 = sub_1648700(v5)[5];
      v7 = *(__int64 **)(a1 + 8);
      if ( *(__int64 **)(a1 + 16) != v7 )
        goto LABEL_3;
      v8 = &v7[*(unsigned int *)(a1 + 28)];
      v9 = *(_DWORD *)(a1 + 28);
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
              goto LABEL_17;
            *v10 = v6;
            --*(_DWORD *)(a1 + 32);
            ++*(_QWORD *)a1;
            goto LABEL_4;
          }
        }
        goto LABEL_4;
      }
LABEL_17:
      if ( v9 < *(_DWORD *)(a1 + 24) )
      {
        *(_DWORD *)(a1 + 28) = v9 + 1;
        *v8 = v6;
        ++*(_QWORD *)a1;
      }
      else
      {
LABEL_3:
        sub_16CCBA0(a1, v6);
      }
      do
LABEL_4:
        v5 = *(_QWORD *)(v5 + 8);
      while ( v5 && (unsigned __int8)(*((_BYTE *)sub_1648700(v5) + 16) - 25) > 9u );
    }
    while ( a3 != v5 );
  }
}
