// Function: sub_2183B30
// Address: 0x2183b30
//
char __fastcall sub_2183B30(__int64 a1, __int64 a2)
{
  char result; // al
  unsigned __int16 v4; // dx
  int v5; // eax
  int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rdi
  int v13; // esi
  __int64 v14; // rdx
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rcx
  char *v20; // rsi
  char *v21; // rax
  __int64 v22; // rcx
  char *v23; // rcx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  int i; // eax
  int v32; // r9d

  if ( !byte_4FD3200 )
    return 0;
  result = a2 == 0 || *(_QWORD *)(a1 + 272) == 0;
  if ( result )
    return 0;
  v4 = **(_WORD **)(a2 + 16);
  if ( v4 <= 0x95u )
  {
    if ( v4 <= 0x93u )
      return result;
LABEL_7:
    v5 = sub_217DB60(a2);
    v6 = v5;
    if ( v5 )
    {
      v7 = *(_QWORD *)(a1 + 248);
      if ( v5 < 0 )
        v8 = *(_QWORD *)(*(_QWORD *)(v7 + 24) + 16LL * (v5 & 0x7FFFFFFF) + 8);
      else
        v8 = *(_QWORD *)(*(_QWORD *)(v7 + 272) + 8LL * (unsigned int)v5);
      while ( 1 )
      {
        if ( !v8 )
          return 0;
        if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 && (*(_BYTE *)(v8 + 4) & 8) == 0 )
          break;
        v8 = *(_QWORD *)(v8 + 32);
      }
LABEL_24:
      v10 = *(_QWORD *)(a1 + 272);
      v11 = *(_DWORD *)(v10 + 256);
      if ( !v11 )
        goto LABEL_21;
      v12 = *(_QWORD *)(v10 + 240);
      v13 = v11 - 1;
      v14 = *(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL);
      v15 = (v11 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v16 = (__int64 *)(v12 + 16LL * v15);
      v17 = *v16;
      if ( v14 != *v16 )
      {
        for ( i = 1; ; i = v32 )
        {
          if ( v17 == -8 )
            goto LABEL_21;
          v32 = i + 1;
          v15 = v13 & (i + v15);
          v16 = (__int64 *)(v12 + 16LL * v15);
          v17 = *v16;
          if ( v14 == *v16 )
            break;
        }
      }
      v18 = v16[1];
      if ( !v18 )
        goto LABEL_21;
      v19 = **(_QWORD **)(v18 + 32);
      v20 = *(char **)(v19 + 72);
      v21 = *(char **)(v19 + 64);
      v22 = (v20 - v21) >> 5;
      v9 = (v20 - v21) >> 3;
      if ( v22 > 0 )
      {
        v23 = &v21[32 * v22];
        while ( v14 != *(_QWORD *)v21 )
        {
          if ( v14 == *((_QWORD *)v21 + 1) )
          {
            v21 += 8;
            goto LABEL_34;
          }
          if ( v14 == *((_QWORD *)v21 + 2) )
          {
            v21 += 16;
            goto LABEL_34;
          }
          if ( v14 == *((_QWORD *)v21 + 3) )
          {
            v21 += 24;
            goto LABEL_34;
          }
          v21 += 32;
          if ( v23 == v21 )
          {
            v9 = (v20 - v21) >> 3;
            goto LABEL_17;
          }
        }
        goto LABEL_34;
      }
LABEL_17:
      if ( v9 != 2 )
      {
        if ( v9 != 3 )
        {
          if ( v9 != 1 )
            goto LABEL_21;
LABEL_20:
          if ( v14 != *(_QWORD *)v21 )
            goto LABEL_21;
          goto LABEL_34;
        }
        if ( v14 == *(_QWORD *)v21 )
        {
LABEL_34:
          if ( v20 != v21 )
          {
            v24 = (_QWORD *)(*(_QWORD *)(v14 + 24) & 0xFFFFFFFFFFFFFFF8LL);
            v25 = v24;
            if ( !v24 )
              BUG();
            v26 = *v24;
            if ( (v26 & 4) == 0 && (*((_BYTE *)v25 + 46) & 4) != 0 )
            {
              do
              {
                v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
                v26 = *(_QWORD *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
              }
              while ( (*(_BYTE *)(v27 + 46) & 4) != 0 );
            }
            v28 = v26 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v28 )
              BUG();
            v29 = *(_QWORD *)v28;
            v30 = v28;
            if ( (*(_QWORD *)v28 & 4) == 0 && (*(_BYTE *)(v28 + 46) & 4) != 0 )
            {
              while ( 1 )
              {
                v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
                  break;
                v29 = *(_QWORD *)v30;
              }
            }
            if ( **(_WORD **)(v30 + 16) == 190 )
            {
              result = sub_21834F0(*(_QWORD *)(a1 + 248), v30, v6);
              if ( result )
                return result;
            }
          }
LABEL_21:
          while ( 1 )
          {
            v8 = *(_QWORD *)(v8 + 32);
            if ( !v8 )
              return 0;
            if ( (*(_BYTE *)(v8 + 3) & 0x10) == 0 && (*(_BYTE *)(v8 + 4) & 8) == 0 )
              goto LABEL_24;
          }
        }
        v21 += 8;
      }
      if ( v14 != *(_QWORD *)v21 )
      {
        v21 += 8;
        goto LABEL_20;
      }
      goto LABEL_34;
    }
    return 0;
  }
  if ( (unsigned __int16)(v4 - 3641) <= 1u )
    goto LABEL_7;
  return result;
}
