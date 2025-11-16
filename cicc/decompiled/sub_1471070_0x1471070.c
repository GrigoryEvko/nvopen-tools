// Function: sub_1471070
// Address: 0x1471070
//
__int64 __fastcall sub_1471070(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // eax
  int v6; // edi
  __int64 v7; // rcx
  __int64 v8; // rsi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 result; // rax
  unsigned int v14; // eax
  __int64 v15; // r12
  __int64 i; // rax
  __int64 v17; // rax
  __int64 v18; // r14
  int v19; // edx
  unsigned int v20; // eax
  __int64 j; // rbx
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  int v25; // r9d

  v2 = *(_QWORD *)(a1 + 64);
  v3 = *(_DWORD *)(v2 + 24);
  if ( !v3 )
    return 0;
  v6 = v3 - 1;
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(v2 + 8);
  v9 = (v3 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( *v10 == v7 )
  {
LABEL_3:
    v12 = v10[1];
    if ( v12 )
    {
      if ( **(_QWORD **)(v12 + 32) == v7 )
      {
        if ( (unsigned __int8)sub_14AEC90(a2) )
        {
          v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          if ( v14 )
          {
            v15 = 0;
            if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
              goto LABEL_28;
LABEL_10:
            for ( i = *(_QWORD *)(a2 - 8); sub_1456C80(a1, **(_QWORD **)(i + 24 * v15)); i = a2 - 24LL * v14 )
            {
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                v17 = *(_QWORD *)(a2 - 8);
              else
                v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              v18 = sub_146F1B0(a1, *(_QWORD *)(v17 + 24 * v15));
              if ( *(_WORD *)(v18 + 24) == 7 )
              {
                v19 = *(_DWORD *)(a2 + 20);
                v20 = v19 & 0xFFFFFFF;
                if ( (v19 & 0xFFFFFFF) != 0 )
                {
                  for ( j = 0; (v19 & 0xFFFFFFFu) > (unsigned int)j; ++j )
                  {
                    if ( (_DWORD)j != (_DWORD)v15 )
                    {
                      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                        v22 = *(_QWORD *)(a2 - 8);
                      else
                        v22 = a2 - 24LL * v20;
                      v23 = sub_146F1B0(a1, *(_QWORD *)(v22 + 24 * j));
                      if ( !sub_146CEE0(a1, v23, *(_QWORD *)(v18 + 48)) )
                        goto LABEL_26;
                      v19 = *(_DWORD *)(a2 + 20);
                    }
                    v20 = v19 & 0xFFFFFFF;
                  }
                }
                result = sub_14AE9E0(a2, *(_QWORD *)(v18 + 48));
                if ( (_BYTE)result )
                  return result;
              }
LABEL_26:
              ++v15;
              v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
              if ( v14 <= (unsigned int)v15 )
                return 0;
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                goto LABEL_10;
LABEL_28:
              ;
            }
          }
        }
      }
    }
  }
  else
  {
    v24 = 1;
    while ( v11 != -8 )
    {
      v25 = v24 + 1;
      v9 = v6 & (v24 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
        goto LABEL_3;
      v24 = v25;
    }
  }
  return 0;
}
