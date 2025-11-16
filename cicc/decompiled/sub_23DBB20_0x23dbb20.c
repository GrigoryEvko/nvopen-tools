// Function: sub_23DBB20
// Address: 0x23dbb20
//
__int64 __fastcall sub_23DBB20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 *i; // rsi
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // r12
  __int64 j; // r13
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // rdx
  __int64 *v18; // rax
  unsigned int v19; // eax

  v6 = a2 + 72;
  v7 = *(_QWORD *)(a2 + 80);
  for ( i = (__int64 *)(a1 + 48); v6 != v7; v7 = *(_QWORD *)(v7 + 8) )
  {
    v9 = *(_QWORD *)(a1 + 24);
    if ( v7 )
    {
      v10 = v7 - 24;
      v11 = (unsigned int)(*(_DWORD *)(v7 + 20) + 1);
      v12 = *(_DWORD *)(v7 + 20) + 1;
    }
    else
    {
      v10 = 0;
      v11 = 0;
      v12 = 0;
    }
    if ( v12 < *(_DWORD *)(v9 + 32) )
    {
      if ( *(_QWORD *)(*(_QWORD *)(v9 + 24) + 8 * v11) )
      {
        v13 = *(_QWORD *)(v10 + 56);
        for ( j = v10 + 48; j != v13; v13 = *(_QWORD *)(v13 + 8) )
        {
          if ( !v13 )
            BUG();
          if ( *(_BYTE *)(v13 - 24) == 67 )
          {
            v15 = *(unsigned int *)(a1 + 40);
            if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
            {
              sub_C8D5F0(a1 + 32, i, v15 + 1, 8u, a5, a6);
              v15 = *(unsigned int *)(a1 + 40);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v15) = v13 - 24;
            ++*(_DWORD *)(a1 + 40);
          }
        }
      }
    }
  }
  v16 = 0;
  while ( 1 )
  {
    v19 = *(_DWORD *)(a1 + 40);
    if ( !v19 )
      break;
    v17 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * v19 - 8);
    *(_DWORD *)(a1 + 40) = v19 - 1;
    *(_QWORD *)(a1 + 80) = v17;
    v18 = (__int64 *)sub_23DB2B0(a1, (__int64)i, v17, v19, a5, a6);
    if ( v18 )
    {
      i = v18;
      v16 = 1;
      sub_23D89F0(a1, v18);
    }
  }
  return v16;
}
