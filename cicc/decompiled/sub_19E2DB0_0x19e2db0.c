// Function: sub_19E2DB0
// Address: 0x19e2db0
//
bool __fastcall sub_19E2DB0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  int v5; // edx
  __int64 v6; // rdi
  int v7; // eax
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  unsigned int v11; // r9d
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // r8
  int v16; // edx
  int v17; // edx
  int v18; // r10d
  int v19; // r10d

  result = 0;
  v5 = *(_DWORD *)(*(_QWORD *)a1 + 2384LL);
  if ( v5 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)a1 + 2368LL);
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v6 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_3:
      v11 = *((_DWORD *)v9 + 2);
      v12 = *(_QWORD *)(a3 + 8);
    }
    else
    {
      v16 = 1;
      while ( v10 != -8 )
      {
        v18 = v16 + 1;
        v8 = v7 & (v16 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_3;
        v16 = v18;
      }
      v12 = *(_QWORD *)(a3 + 8);
      v11 = 0;
    }
    v13 = v7 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v14 = (__int64 *)(v6 + 16LL * v13);
    v15 = *v14;
    if ( v12 == *v14 )
    {
      return v11 < *((_DWORD *)v14 + 2);
    }
    else
    {
      v17 = 1;
      while ( v15 != -8 )
      {
        v19 = v17 + 1;
        v13 = v7 & (v17 + v13);
        v14 = (__int64 *)(v6 + 16LL * v13);
        v15 = *v14;
        if ( v12 == *v14 )
          return v11 < *((_DWORD *)v14 + 2);
        v17 = v19;
      }
      return 0;
    }
  }
  return result;
}
