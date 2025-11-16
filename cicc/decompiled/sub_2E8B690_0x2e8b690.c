// Function: sub_2E8B690
// Address: 0x2e8b690
//
char __fastcall sub_2E8B690(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int64 a5)
{
  int v7; // eax
  __int64 v8; // rbx
  _QWORD *v9; // rdi
  __int64 v10; // r12
  unsigned __int8 v11; // al
  char v12; // r9
  unsigned int v13; // ecx
  __int16 *v14; // rsi
  int v15; // edx
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rdx
  unsigned int v20; // [rsp+4h] [rbp-3Ch]

  v7 = *(_DWORD *)(a1 + 40);
  v8 = *(_QWORD *)(a1 + 32);
  v9 = (_QWORD *)a2;
  v10 = v8 + 40LL * (v7 & 0xFFFFFF);
  while ( v10 != v8 )
  {
    if ( !*(_BYTE *)v8 )
    {
      v11 = *(_BYTE *)(v8 + 3);
      if ( (v11 & 0x10) != 0 )
      {
        a5 = *(unsigned int *)(v8 + 8);
        if ( (unsigned int)(a5 - 1) > 0x3FFFFFFE )
        {
          if ( (((v11 & 0x10) != 0) & (v11 >> 6)) == 0 )
          {
            if ( (a5 & 0x80000000) != 0LL )
              v17 = *(_QWORD *)(v9[7] + 16 * (a5 & 0x7FFFFFFF) + 8);
            else
              v17 = *(_QWORD *)(v9[38] + 8 * a5);
            while ( v17 )
            {
              if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
              {
                a4 = *(_QWORD *)(v17 + 16);
LABEL_24:
                if ( a1 == a4 )
                {
                  while ( 1 )
                  {
                    v17 = *(_QWORD *)(v17 + 32);
                    if ( !v17 )
                      goto LABEL_4;
                    if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0
                      && (*(_BYTE *)(v17 + 4) & 8) == 0
                      && a4 != *(_QWORD *)(v17 + 16) )
                    {
                      a4 = *(_QWORD *)(v17 + 16);
                      goto LABEL_24;
                    }
                  }
                }
                return 0;
              }
              v17 = *(_QWORD *)(v17 + 32);
            }
          }
        }
        else
        {
          a2 = (unsigned int)a5;
          v20 = *(_DWORD *)(v8 + 8);
          v12 = sub_2EBF3A0(v9, (unsigned int)a5);
          if ( !v12 )
          {
            a5 = v20;
            if ( !a3 )
              return 0;
            v13 = *(_DWORD *)(*(_QWORD *)(*a3 + 8LL) + 24LL * v20 + 16) & 0xFFF;
            v14 = (__int16 *)(*(_QWORD *)(*a3 + 56LL)
                            + 2LL * (*(_DWORD *)(*(_QWORD *)(*a3 + 8LL) + 24LL * v20 + 16) >> 12));
            do
            {
              if ( !v14 )
                break;
              if ( (*(_QWORD *)(a3[1] + 8LL * (v13 >> 6)) & (1LL << v13)) != 0 )
                return v12;
              v15 = *v14++;
              v13 += v15;
            }
            while ( (_WORD)v15 );
            a2 = v9[48];
            a4 = v20;
            if ( (*(_QWORD *)(a2 + 8LL * (v20 >> 6)) & (1LL << v20)) != 0 )
              return 0;
          }
        }
      }
    }
LABEL_4:
    v8 += 40;
  }
  v18 = *(unsigned __int16 *)(a1 + 68);
  v19 = (unsigned int)(v18 - 1);
  if ( (unsigned int)v19 <= 1 )
    return 0;
  v12 = 1;
  if ( (unsigned int)(v18 - 22) <= 1 )
    return v12;
  return sub_2E8B630(a1, a2, v19, a4, (_QWORD *)a5);
}
