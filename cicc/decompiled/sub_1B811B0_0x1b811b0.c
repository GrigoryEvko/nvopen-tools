// Function: sub_1B811B0
// Address: 0x1b811b0
//
void __fastcall sub_1B811B0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v4; // r12
  __int64 *v6; // r8
  __int64 *v7; // r9
  __int64 v8; // rsi
  __int64 *v9; // rdi
  unsigned int v10; // r10d
  __int64 *v11; // rax
  __int64 *v12; // rcx

  if ( a2 != a3 )
  {
    v4 = a2;
    v6 = *(__int64 **)(a1 + 16);
    v7 = *(__int64 **)(a1 + 8);
    do
    {
LABEL_5:
      v8 = *v4;
      if ( v6 != v7 )
        goto LABEL_3;
      v9 = &v6[*(unsigned int *)(a1 + 28)];
      v10 = *(_DWORD *)(a1 + 28);
      if ( v9 != v6 )
      {
        v11 = v6;
        v12 = 0;
        while ( v8 != *v11 )
        {
          if ( *v11 == -2 )
            v12 = v11;
          if ( v9 == ++v11 )
          {
            if ( !v12 )
              goto LABEL_15;
            ++v4;
            *v12 = v8;
            v6 = *(__int64 **)(a1 + 16);
            --*(_DWORD *)(a1 + 32);
            v7 = *(__int64 **)(a1 + 8);
            ++*(_QWORD *)a1;
            if ( a3 != v4 )
              goto LABEL_5;
            return;
          }
        }
        goto LABEL_4;
      }
LABEL_15:
      if ( v10 < *(_DWORD *)(a1 + 24) )
      {
        *(_DWORD *)(a1 + 28) = v10 + 1;
        *v9 = v8;
        v7 = *(__int64 **)(a1 + 8);
        ++*(_QWORD *)a1;
        v6 = *(__int64 **)(a1 + 16);
      }
      else
      {
LABEL_3:
        sub_16CCBA0(a1, v8);
        v6 = *(__int64 **)(a1 + 16);
        v7 = *(__int64 **)(a1 + 8);
      }
LABEL_4:
      ++v4;
    }
    while ( a3 != v4 );
  }
}
