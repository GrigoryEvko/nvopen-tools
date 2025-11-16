// Function: sub_31BA7D0
// Address: 0x31ba7d0
//
char __fastcall sub_31BA7D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  bool v5; // al
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // rdi
  int v17; // edx
  int v18; // r9d
  int v19; // edx
  int v20; // r9d

  v4 = sub_318E5D0(a2);
  v5 = sub_318B630(v4);
  if ( v4 )
  {
    if ( v5 )
    {
      v6 = *(unsigned int *)(a1 + 24);
      v7 = *(_QWORD *)(a1 + 8);
      if ( (_DWORD)v6 )
      {
        v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v4 == *v9 )
        {
LABEL_5:
          if ( v9 != (__int64 *)(v7 + 16 * v6) )
          {
            v11 = v9[1];
            if ( v11 )
              --*(_DWORD *)(v11 + 20);
          }
        }
        else
        {
          v19 = 1;
          while ( v10 != -4096 )
          {
            v20 = v19 + 1;
            v8 = (v6 - 1) & (v19 + v8);
            v9 = (__int64 *)(v7 + 16LL * v8);
            v10 = *v9;
            if ( v4 == *v9 )
              goto LABEL_5;
            v19 = v20;
          }
        }
      }
    }
  }
  LOBYTE(v12) = sub_318B630(a3);
  if ( a3 )
  {
    if ( (_BYTE)v12 )
    {
      v12 = *(unsigned int *)(a1 + 24);
      v13 = *(_QWORD *)(a1 + 8);
      if ( (_DWORD)v12 )
      {
        v14 = (v12 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v15 = (__int64 *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( a3 == *v15 )
        {
LABEL_12:
          v12 = v13 + 16 * v12;
          if ( v15 != (__int64 *)v12 )
          {
            v12 = v15[1];
            if ( v12 )
              ++*(_DWORD *)(v12 + 20);
          }
        }
        else
        {
          v17 = 1;
          while ( v16 != -4096 )
          {
            v18 = v17 + 1;
            v14 = (v12 - 1) & (v17 + v14);
            v15 = (__int64 *)(v13 + 16LL * v14);
            v16 = *v15;
            if ( a3 == *v15 )
              goto LABEL_12;
            v17 = v18;
          }
        }
      }
    }
  }
  return v12;
}
