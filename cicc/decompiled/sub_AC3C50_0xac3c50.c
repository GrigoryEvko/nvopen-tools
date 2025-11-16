// Function: sub_AC3C50
// Address: 0xac3c50
//
void __fastcall __noreturn sub_AC3C50(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r13
  int v7; // eax
  __int64 v8; // rsi
  int v9; // edx
  unsigned int v10; // eax
  __int64 *v11; // rbx
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r13
  int v17; // eax
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  __int64 *v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // r12
  int v24; // r8d
  int v25; // r8d

  v3 = *a1;
  if ( v3 == 12 )
  {
    v14 = sub_BD5C60(a1, a2, a3);
    v15 = *((_QWORD *)a1 + 1);
    v16 = *(_QWORD *)v14;
    v17 = *(_DWORD *)(*(_QWORD *)v14 + 1928LL);
    v18 = *(_QWORD *)(v16 + 1912);
    if ( v17 )
    {
      v19 = v17 - 1;
      v20 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( v15 == *v21 )
      {
LABEL_9:
        v23 = v21[1];
        if ( v23 )
        {
          sub_BD7260(v21[1]);
          sub_BD2DD0(v23);
        }
        *v21 = -8192;
        --*(_DWORD *)(v16 + 1920);
        ++*(_DWORD *)(v16 + 1924);
        BUG();
      }
      v25 = 1;
      while ( v22 != -4096 )
      {
        v20 = v19 & (v25 + v20);
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( v15 == *v21 )
          goto LABEL_9;
        ++v25;
      }
    }
  }
  else if ( v3 == 13 )
  {
    v4 = sub_BD5C60(a1, a2, a3);
    v5 = *((_QWORD *)a1 + 1);
    v6 = *(_QWORD *)v4;
    v7 = *(_DWORD *)(*(_QWORD *)v4 + 1960LL);
    v8 = *(_QWORD *)(v6 + 1944);
    if ( v7 )
    {
      v9 = v7 - 1;
      v10 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v5 == *v11 )
      {
LABEL_5:
        v13 = v11[1];
        if ( v13 )
        {
          sub_BD7260(v11[1]);
          sub_BD2DD0(v13);
        }
        *v11 = -8192;
        --*(_DWORD *)(v6 + 1952);
        ++*(_DWORD *)(v6 + 1956);
      }
      else
      {
        v24 = 1;
        while ( v12 != -4096 )
        {
          v10 = v9 & (v24 + v10);
          v11 = (__int64 *)(v8 + 16LL * v10);
          v12 = *v11;
          if ( v5 == *v11 )
            goto LABEL_5;
          ++v24;
        }
      }
    }
  }
  BUG();
}
