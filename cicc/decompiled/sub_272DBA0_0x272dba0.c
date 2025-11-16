// Function: sub_272DBA0
// Address: 0x272dba0
//
__int64 __fastcall sub_272DBA0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax

  v5 = a2;
  if ( *(_BYTE *)a1 == 84 )
  {
    v6 = *(_QWORD *)(a1 - 8);
    v7 = 32LL * *(unsigned int *)(a1 + 72);
    v8 = *(_QWORD *)(v6 + 8LL * a2 + v7);
    if ( a2 )
    {
      v9 = 0;
      v10 = v6 + v7;
      while ( v8 != *(_QWORD *)(v10 + 8 * v9) )
      {
        if ( v5 == ++v9 )
          goto LABEL_14;
      }
      v11 = *(_QWORD *)(v6 + 32 * v9);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v12 = 32 * v5 + v6;
        if ( *(_QWORD *)v12 )
          goto LABEL_8;
      }
      else
      {
        v12 = 32 * v5 + a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v12 )
        {
LABEL_8:
          v13 = *(_QWORD *)(v12 + 8);
          **(_QWORD **)(v12 + 16) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
        }
      }
      *(_QWORD *)v12 = v11;
      result = 0;
      if ( v11 )
      {
        v15 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v12 + 8) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = v12 + 8;
        *(_QWORD *)(v12 + 16) = v11 + 16;
        *(_QWORD *)(v11 + 16) = v12;
        return 0;
      }
      return result;
    }
  }
LABEL_14:
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v16 = *(_QWORD *)(a1 - 8);
  else
    v16 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v17 = 32 * v5 + v16;
  if ( *(_QWORD *)v17 )
  {
    v18 = *(_QWORD *)(v17 + 8);
    **(_QWORD **)(v17 + 16) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = *(_QWORD *)(v17 + 16);
  }
  *(_QWORD *)v17 = a3;
  result = 1;
  if ( a3 )
  {
    v19 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v17 + 8) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = v17 + 8;
    *(_QWORD *)(v17 + 16) = a3 + 16;
    result = 1;
    *(_QWORD *)(a3 + 16) = v17;
  }
  return result;
}
