// Function: sub_2A00560
// Address: 0x2a00560
//
__int64 __fastcall sub_2A00560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r11
  __int64 v10; // rcx
  unsigned int v11; // r9d
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 result; // rax

  v7 = sub_AA5930(*(_QWORD *)(a2 + 8));
  if ( v8 != v7 )
  {
    v9 = v8;
    v10 = v7;
    v11 = 0;
    do
    {
      v12 = v11++;
      v13 = *(_QWORD *)(*(_QWORD *)(a4 + 16) + 8 * v12);
      if ( (*(_DWORD *)(v10 + 4) & 0x7FFFFFF) != 0 )
      {
        v14 = 0;
        v15 = 8LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
        do
        {
          while ( 1 )
          {
            v16 = *(_QWORD *)(v10 - 8);
            if ( a3 == *(_QWORD *)(v16 + 32LL * *(unsigned int *)(v10 + 72) + v14) )
            {
              v17 = v16 + 4 * v14;
              if ( *(_QWORD *)v17 )
              {
                v18 = *(_QWORD *)(v17 + 8);
                **(_QWORD **)(v17 + 16) = v18;
                if ( v18 )
                  *(_QWORD *)(v18 + 16) = *(_QWORD *)(v17 + 16);
              }
              *(_QWORD *)v17 = v13;
              if ( v13 )
                break;
            }
            v14 += 8;
            if ( v15 == v14 )
              goto LABEL_14;
          }
          v19 = *(_QWORD *)(v13 + 16);
          *(_QWORD *)(v17 + 8) = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = v17 + 8;
          v14 += 8;
          *(_QWORD *)(v17 + 16) = v13 + 16;
          *(_QWORD *)(v13 + 16) = v17;
        }
        while ( v15 != v14 );
      }
LABEL_14:
      v20 = *(_QWORD *)(v10 + 32);
      if ( !v20 )
        BUG();
      v10 = 0;
      if ( *(_BYTE *)(v20 - 24) == 84 )
        v10 = v20 - 24;
    }
    while ( v9 != v10 );
  }
  result = *(_QWORD *)(a4 + 40);
  *(_QWORD *)(a2 + 56) = result;
  return result;
}
