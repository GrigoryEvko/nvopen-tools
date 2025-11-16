// Function: sub_2FF90F0
// Address: 0x2ff90f0
//
__int64 __fastcall sub_2FF90F0(__int64 a1, int a2, unsigned int a3, _DWORD *a4)
{
  __int64 v5; // rax
  __int64 v7; // rax
  unsigned int v8; // r11d
  unsigned int v9; // ebx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r12
  unsigned int v13; // r10d
  __int64 *v14; // rsi
  __int64 v15; // r14
  int v17; // esi
  int v18; // r15d

  *a4 = 0;
  v5 = *(_QWORD *)(a1 + 32);
  if ( a2 < 0 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v5 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
    if ( v7 )
      goto LABEL_3;
  }
  else
  {
    v7 = *(_QWORD *)(*(_QWORD *)(v5 + 304) + 8LL * (unsigned int)a2);
    if ( v7 )
    {
LABEL_3:
      v8 = a3;
      v9 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)(v7 + 16);
          if ( *(_QWORD *)(v10 + 24) == *(_QWORD *)(a1 + 72) && (unsigned __int16)(*(_WORD *)(v10 + 68) - 14) > 1u )
          {
            v11 = *(unsigned int *)(a1 + 104);
            v12 = *(_QWORD *)(a1 + 88);
            if ( (_DWORD)v11 )
            {
              v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
              v14 = (__int64 *)(v12 + 16LL * v13);
              v15 = *v14;
              if ( v10 != *v14 )
              {
                v17 = 1;
                while ( v15 != -4096 )
                {
                  v18 = v17 + 1;
                  v13 = (v11 - 1) & (v17 + v13);
                  v14 = (__int64 *)(v12 + 16LL * v13);
                  v15 = *v14;
                  if ( v10 == *v14 )
                    goto LABEL_9;
                  v17 = v18;
                }
                goto LABEL_4;
              }
LABEL_9:
              if ( v14 != (__int64 *)(v12 + 16 * v11) )
                break;
            }
          }
LABEL_4:
          v7 = *(_QWORD *)(v7 + 32);
          if ( !v7 )
            goto LABEL_14;
        }
        LODWORD(v10) = *((_DWORD *)v14 + 2);
        if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
        {
          if ( v9 < (unsigned int)v10 )
          {
            *a4 = v10;
            v9 = v10;
          }
          goto LABEL_4;
        }
        v7 = *(_QWORD *)(v7 + 32);
        if ( v8 > (unsigned int)v10 )
          v8 = *((_DWORD *)v14 + 2);
        if ( !v7 )
        {
LABEL_14:
          LOBYTE(v7) = v8 <= v9;
          LOBYTE(v10) = a3 <= v8;
          return (unsigned int)v10 | (unsigned int)v7;
        }
      }
    }
  }
  return 1;
}
