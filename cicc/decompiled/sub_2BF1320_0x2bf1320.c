// Function: sub_2BF1320
// Address: 0x2bf1320
//
__int64 __fastcall sub_2BF1320(__int64 a1, __int64 a2)
{
  __int64 *v2; // rsi
  __int64 *v3; // r9
  int v4; // r10d
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r11
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // r12
  unsigned int v14; // eax
  int v15; // r13d
  unsigned int v16; // ebx
  int *v17; // r11
  int v18; // edx
  unsigned int v19; // ebx
  _BOOL8 v21; // rax

  v2 = *(__int64 **)(a1 + 88);
  v3 = &v2[*(unsigned int *)(a1 + 96)];
  if ( v3 == v2 )
    goto LABEL_47;
  v4 = (BYTE4(a2) == 0) + 37 * a2 - 1;
LABEL_3:
  while ( 2 )
  {
    v5 = *v2;
    if ( *(_DWORD *)(*v2 + 64) )
    {
      v12 = *(_DWORD *)(v5 + 72);
      v13 = *(_QWORD *)(v5 + 56);
      if ( !v12 )
        goto LABEL_15;
      v14 = v12 - 1;
      v15 = 1;
      v16 = v4 & v14;
      v17 = (int *)(v13 + 8LL * (v4 & v14));
      v18 = *v17;
      if ( (_DWORD)a2 == *v17 )
        goto LABEL_21;
      do
      {
        do
        {
          if ( v18 == -1 && *((_BYTE *)v17 + 4) )
            goto LABEL_15;
          v19 = v15 + v16;
          ++v15;
          v16 = v14 & v19;
          v17 = (int *)(v13 + 8LL * v16);
          v18 = *v17;
        }
        while ( (_DWORD)a2 != *v17 );
LABEL_21:
        ;
      }
      while ( BYTE4(a2) != *((_BYTE *)v17 + 4) );
    }
    else
    {
      v6 = *(_QWORD *)(v5 + 80);
      v7 = 8LL * *(unsigned int *)(v5 + 88);
      v8 = v6 + v7;
      v9 = v7 >> 3;
      v10 = v7 >> 5;
      if ( !v10 )
        goto LABEL_12;
      v11 = v6 + 32 * v10;
      while ( 1 )
      {
        if ( (_DWORD)a2 == *(_DWORD *)v6 && *(_BYTE *)(v6 + 4) == BYTE4(a2) )
          goto LABEL_24;
        if ( (_DWORD)a2 == *(_DWORD *)(v6 + 8) && *(_BYTE *)(v6 + 12) == BYTE4(a2) )
        {
          v21 = v8 != v6 + 8;
          goto LABEL_25;
        }
        if ( (_DWORD)a2 == *(_DWORD *)(v6 + 16) && *(_BYTE *)(v6 + 20) == BYTE4(a2) )
        {
          v21 = v8 != v6 + 16;
          goto LABEL_25;
        }
        if ( (_DWORD)a2 == *(_DWORD *)(v6 + 24) && *(_BYTE *)(v6 + 28) == BYTE4(a2) )
          break;
        v6 += 32;
        if ( v11 == v6 )
        {
          v9 = (v8 - v6) >> 3;
LABEL_12:
          if ( v9 != 2 )
          {
            if ( v9 != 3 )
            {
              if ( v9 != 1 )
                goto LABEL_15;
LABEL_38:
              if ( (_DWORD)a2 == *(_DWORD *)v6 && *(_BYTE *)(v6 + 4) == BYTE4(a2) )
                goto LABEL_24;
LABEL_15:
              if ( v3 == ++v2 )
LABEL_47:
                BUG();
              goto LABEL_3;
            }
            if ( (_DWORD)a2 == *(_DWORD *)v6 && *(_BYTE *)(v6 + 4) == BYTE4(a2) )
              goto LABEL_24;
            v6 += 8;
          }
          if ( (_DWORD)a2 != *(_DWORD *)v6 || *(_BYTE *)(v6 + 4) != BYTE4(a2) )
          {
            v6 += 8;
            goto LABEL_38;
          }
LABEL_24:
          v21 = v8 != v6;
          goto LABEL_25;
        }
      }
      v21 = v8 != v6 + 24;
LABEL_25:
      if ( !v21 )
      {
        if ( v3 == ++v2 )
          goto LABEL_47;
        continue;
      }
    }
    return *v2;
  }
}
