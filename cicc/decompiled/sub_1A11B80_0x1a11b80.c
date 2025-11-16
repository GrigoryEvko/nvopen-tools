// Function: sub_1A11B80
// Address: 0x1a11b80
//
__int64 __fastcall sub_1A11B80(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 v7; // rbx
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rsi
  int v13; // edi
  __int64 v14; // r9
  int v15; // edi
  unsigned int v16; // r10d
  __int64 v17; // r8
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rsi
  unsigned int v21; // eax
  unsigned __int64 v22; // rcx
  _QWORD *v23; // rcx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // [rsp+Ch] [rbp-44h]
  __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a1;
  v3 = (__int64)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 13 )
    return sub_1A11830(a1, (__int64)a2);
  result = *sub_1A10F60(a1, (__int64)a2) ^ 6LL;
  if ( (result & 6) == 0 )
    return result;
  v5 = a2[5] & 0xFFFFFFF;
  if ( (unsigned int)v5 > 0x40 )
  {
LABEL_21:
    a2 = (_DWORD *)v3;
    a1 = v2;
    return sub_1A11830(a1, (__int64)a2);
  }
  if ( (_DWORD)v5 )
  {
    v6 = 8 * v5;
    v7 = 0;
    v8 = 0;
    do
    {
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        v9 = *(_QWORD *)(v3 - 8);
      else
        v9 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
      result = (__int64)sub_1A10F60(v2, *(_QWORD *)(v9 + 3 * v7));
      v10 = *(_QWORD *)result;
      v11 = (*(__int64 *)result >> 1) & 3;
      if ( ((*(__int64 *)result >> 1) & 3) != 0 )
      {
        if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
        {
          v12 = *(_QWORD *)(v3 - 8);
        }
        else
        {
          result = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
          v12 = v3 - result;
        }
        v13 = *(_DWORD *)(v2 + 2424);
        if ( v13 )
        {
          v14 = *(_QWORD *)(v3 + 40);
          v15 = v13 - 1;
          v28 = 1;
          v16 = (unsigned int)v14 >> 9;
          v17 = *(_QWORD *)(v7 + v12 + 24LL * *(unsigned int *)(v3 + 56) + 8);
          v18 = (((v16 ^ ((unsigned int)v14 >> 4)
                 | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(v16 ^ ((unsigned int)v14 >> 4)) << 32)) >> 22)
              ^ ((v16 ^ ((unsigned int)v14 >> 4)
                | ((unsigned __int64)(((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(v16 ^ ((unsigned int)v14 >> 4)) << 32));
          v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
              ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
          for ( result = v15 & ((unsigned int)((v19 - 1 - (v19 << 27)) >> 31) ^ ((_DWORD)v19 - 1 - ((_DWORD)v19 << 27)));
                ;
                result = v15 & v21 )
          {
            v20 = (_QWORD *)(*(_QWORD *)(v2 + 2408) + 16LL * (unsigned int)result);
            if ( v17 == *v20 && v14 == v20[1] )
              break;
            if ( *v20 == -8 && v20[1] == -8 )
              goto LABEL_24;
            v21 = v28 + result;
            ++v28;
          }
          if ( v11 == 3 )
            goto LABEL_21;
          v22 = v10 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v8 )
          {
            if ( v8 != v22 )
              goto LABEL_21;
          }
          else
          {
            v8 = v22;
          }
        }
      }
LABEL_24:
      v7 += 8;
    }
    while ( v7 != v6 );
    if ( v8 )
    {
      v29[0] = v3;
      v23 = sub_1A10690(v2 + 120, v29);
      result = v23[1];
      v26 = (result >> 1) & 3;
      if ( v26 != 1 && v26 != 3 )
      {
        if ( (_DWORD)v26 )
        {
          if ( v8 == (result & 0xFFFFFFFFFFFFFFF8LL) )
            return result;
          v27 = result | 6;
          v23[1] = v27;
        }
        else
        {
          v27 = v8 | v23[1] & 1LL | 2;
          v23[1] = v27;
        }
        if ( (((unsigned __int8)v27 ^ 6) & 6) != 0 )
        {
          result = *(unsigned int *)(v2 + 1352);
          if ( (unsigned int)result >= *(_DWORD *)(v2 + 1356) )
          {
            sub_16CD150(v2 + 1344, (const void *)(v2 + 1360), 0, 8, v24, v25);
            result = *(unsigned int *)(v2 + 1352);
          }
          *(_QWORD *)(*(_QWORD *)(v2 + 1344) + 8 * result) = v3;
          ++*(_DWORD *)(v2 + 1352);
        }
        else
        {
          result = *(unsigned int *)(v2 + 824);
          if ( (unsigned int)result >= *(_DWORD *)(v2 + 828) )
          {
            sub_16CD150(v2 + 816, (const void *)(v2 + 832), 0, 8, v24, v25);
            result = *(unsigned int *)(v2 + 824);
          }
          *(_QWORD *)(*(_QWORD *)(v2 + 816) + 8 * result) = v3;
          ++*(_DWORD *)(v2 + 824);
        }
      }
    }
  }
  return result;
}
