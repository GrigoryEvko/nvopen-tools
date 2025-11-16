// Function: sub_1A11EB0
// Address: 0x1a11eb0
//
signed __int64 __fastcall sub_1A11EB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  signed __int64 result; // rax
  signed __int64 v7; // rbx
  unsigned int v8; // esi
  __int64 v9; // rdi
  __int64 *v10; // r9
  __int64 *v11; // r8
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // eax
  int v17; // esi
  unsigned int v18; // eax
  int v19; // edx
  __int64 *v20; // rcx
  __int64 v21; // rdi
  signed __int64 v22; // rax
  int v23; // r11d
  int v24; // eax
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // r14d
  __int64 v29; // rsi
  int v30; // r10d

  v4 = *sub_1A10F60(a1, *(_QWORD *)(a2 - 24));
  v5 = (v4 >> 1) & 3;
  if ( v5 == 3 )
    return sub_1A11830(a1, a2);
  result = (unsigned int)(v5 - 1);
  if ( (unsigned int)result <= 1 )
  {
    result = sub_14D7A60(
               (unsigned int)*(unsigned __int8 *)(a2 + 16) - 24,
               v4 & 0xFFFFFFFFFFFFFFF8LL,
               *(_QWORD *)a2,
               *(_BYTE **)a1);
    v7 = result;
    if ( *(_BYTE *)(result + 16) != 9 )
    {
      v8 = *(_DWORD *)(a1 + 144);
      v9 = a1 + 120;
      if ( v8 )
      {
        v10 = *(__int64 **)(a1 + 128);
        LODWORD(v11) = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v12 = &v10[2 * (unsigned int)v11];
        v13 = *v12;
        if ( a2 == *v12 )
        {
LABEL_6:
          result = v12[1];
          v14 = (result >> 1) & 3;
          if ( v14 == 1 || v14 == 3 )
            return result;
          if ( (_DWORD)v14 )
          {
            if ( v7 == (result & 0xFFFFFFFFFFFFFFF8LL) )
              return result;
            v15 = result | 6;
            v12[1] = v15;
            goto LABEL_11;
          }
          v22 = result & 1;
          v20 = v12;
LABEL_23:
          v15 = v7 | v22 | 2;
          v20[1] = v15;
LABEL_11:
          if ( (((unsigned __int8)v15 ^ 6) & 6) != 0 )
          {
            result = *(unsigned int *)(a1 + 1352);
            if ( (unsigned int)result >= *(_DWORD *)(a1 + 1356) )
            {
              sub_16CD150(a1 + 1344, (const void *)(a1 + 1360), 0, 8, (int)v11, (int)v10);
              result = *(unsigned int *)(a1 + 1352);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 1344) + 8 * result) = a2;
            ++*(_DWORD *)(a1 + 1352);
          }
          else
          {
            result = *(unsigned int *)(a1 + 824);
            if ( (unsigned int)result >= *(_DWORD *)(a1 + 828) )
            {
              sub_16CD150(a1 + 816, (const void *)(a1 + 832), 0, 8, (int)v11, (int)v10);
              result = *(unsigned int *)(a1 + 824);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 816) + 8 * result) = a2;
            ++*(_DWORD *)(a1 + 824);
          }
          return result;
        }
        v23 = 1;
        v20 = 0;
        while ( v13 != -8 )
        {
          if ( v13 == -16 && !v20 )
            v20 = v12;
          LODWORD(v11) = (v8 - 1) & (v23 + (_DWORD)v11);
          v12 = &v10[2 * (unsigned int)v11];
          v13 = *v12;
          if ( a2 == *v12 )
            goto LABEL_6;
          ++v23;
        }
        v24 = *(_DWORD *)(a1 + 136);
        if ( !v20 )
          v20 = v12;
        ++*(_QWORD *)(a1 + 120);
        v19 = v24 + 1;
        if ( 4 * (v24 + 1) < 3 * v8 )
        {
          LODWORD(v11) = v8 >> 3;
          if ( v8 - *(_DWORD *)(a1 + 140) - v19 > v8 >> 3 )
          {
LABEL_20:
            *(_DWORD *)(a1 + 136) = v19;
            if ( *v20 != -8 )
              --*(_DWORD *)(a1 + 140);
            *v20 = a2;
            v22 = 0;
            v20[1] = 0;
            goto LABEL_23;
          }
          sub_1A0FE70(v9, v8);
          v25 = *(_DWORD *)(a1 + 144);
          if ( v25 )
          {
            v26 = v25 - 1;
            LODWORD(v10) = 1;
            v11 = 0;
            v27 = *(_QWORD *)(a1 + 128);
            v28 = v26 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v19 = *(_DWORD *)(a1 + 136) + 1;
            v20 = (__int64 *)(v27 + 16LL * v28);
            v29 = *v20;
            if ( a2 != *v20 )
            {
              while ( v29 != -8 )
              {
                if ( !v11 && v29 == -16 )
                  v11 = v20;
                v28 = v26 & ((_DWORD)v10 + v28);
                v20 = (__int64 *)(v27 + 16LL * v28);
                v29 = *v20;
                if ( a2 == *v20 )
                  goto LABEL_20;
                LODWORD(v10) = (_DWORD)v10 + 1;
              }
              if ( v11 )
                v20 = v11;
            }
            goto LABEL_20;
          }
LABEL_60:
          ++*(_DWORD *)(a1 + 136);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 120);
      }
      sub_1A0FE70(v9, 2 * v8);
      v16 = *(_DWORD *)(a1 + 144);
      if ( v16 )
      {
        v17 = v16 - 1;
        v11 = *(__int64 **)(a1 + 128);
        v18 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v19 = *(_DWORD *)(a1 + 136) + 1;
        v20 = &v11[2 * v18];
        v21 = *v20;
        if ( a2 != *v20 )
        {
          v30 = 1;
          v10 = 0;
          while ( v21 != -8 )
          {
            if ( v21 == -16 && !v10 )
              v10 = v20;
            v18 = v17 & (v30 + v18);
            v20 = &v11[2 * v18];
            v21 = *v20;
            if ( a2 == *v20 )
              goto LABEL_20;
            ++v30;
          }
          if ( v10 )
            v20 = v10;
        }
        goto LABEL_20;
      }
      goto LABEL_60;
    }
  }
  return result;
}
