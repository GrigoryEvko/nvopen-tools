// Function: sub_1B76840
// Address: 0x1b76840
//
__int64 __fastcall sub_1B76840(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rax
  int v19; // eax
  int v20; // r10d

  v9 = *(_QWORD *)(a2 + 24) + 16LL * *(unsigned int *)(a2 + 16);
  v10 = *(_QWORD *)v9;
  if ( *(_BYTE *)(*(_QWORD *)v9 + 64LL) )
  {
    v13 = *(unsigned int *)(v10 + 56);
    if ( (_DWORD)v13 )
    {
      v14 = *(_QWORD *)(v10 + 40);
      v15 = (v13 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( a3 == *v16 )
      {
LABEL_13:
        if ( v16 != (__int64 *)(v14 + 16 * v13) )
        {
          v18 = v16[1];
          *(_BYTE *)(a1 + 8) = 1;
          *(_QWORD *)a1 = v18;
          return a1;
        }
      }
      else
      {
        v19 = 1;
        while ( v17 != -4 )
        {
          v20 = v19 + 1;
          v15 = (v13 - 1) & (v19 + v15);
          v16 = (__int64 *)(v14 + 16LL * v15);
          v17 = *v16;
          if ( a3 == *v16 )
            goto LABEL_13;
          v19 = v20;
        }
      }
    }
  }
  if ( !*(_BYTE *)a3 || (*(_BYTE *)a2 & 1) != 0 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = a3;
  }
  else if ( *(_BYTE *)a3 == 1 )
  {
    *(_BYTE *)(v10 + 73) = 0;
    v12 = sub_1B75C50(a2, *(_QWORD *)(a3 + 136), a4, a5, a6);
    if ( v12 == *(_QWORD *)(a3 + 136) )
    {
      v12 = a3;
    }
    else if ( v12 )
    {
      v12 = (__int64)sub_1624210(v12);
    }
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v12;
    *(_BYTE *)(v10 + 73) = 1;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
  }
  return a1;
}
