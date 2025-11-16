// Function: sub_1213420
// Address: 0x1213420
//
__int64 __fastcall sub_1213420(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  unsigned int v4; // r13d
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rsi
  size_t v9; // r15
  size_t v10; // r10
  size_t v11; // rdx
  unsigned int v12; // eax
  size_t v13; // rbx
  size_t v14; // r13
  size_t v15; // rdx
  int v16; // eax
  __int64 v17; // rbx
  __int64 v19; // r15
  size_t v20; // [rsp+8h] [rbp-38h]

  v2 = a1 + 8;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return a1 + 8;
  v4 = *(_DWORD *)a2;
  while ( 1 )
  {
    if ( *(_DWORD *)(v3 + 32) != v4 )
    {
      LOBYTE(v6) = *(_DWORD *)(v3 + 32) < (int)v4;
      goto LABEL_4;
    }
    if ( v4 <= 1 )
      break;
    v9 = *(_QWORD *)(v3 + 72);
    v10 = *(_QWORD *)(a2 + 40);
    v11 = v10;
    if ( v9 <= v10 )
      v11 = *(_QWORD *)(v3 + 72);
    if ( v11 )
    {
      v20 = *(_QWORD *)(a2 + 40);
      v12 = memcmp(*(const void **)(v3 + 64), *(const void **)(a2 + 32), v11);
      v10 = v20;
      if ( v12 )
      {
        v6 = v12 >> 31;
        goto LABEL_4;
      }
    }
    v19 = v9 - v10;
    if ( v19 < 0x80000000LL )
    {
      if ( v19 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        LOBYTE(v6) = (int)v19 < 0;
        goto LABEL_4;
      }
      v8 = *(_QWORD *)(v3 + 24);
LABEL_5:
      v7 = v8;
      goto LABEL_6;
    }
    v7 = *(_QWORD *)(v3 + 16);
    v2 = v3;
LABEL_6:
    if ( !v7 )
      goto LABEL_16;
LABEL_7:
    v3 = v7;
  }
  LOBYTE(v6) = *(_DWORD *)(v3 + 48) < *(_DWORD *)(a2 + 16);
LABEL_4:
  v7 = *(_QWORD *)(v3 + 16);
  v8 = *(_QWORD *)(v3 + 24);
  if ( (_BYTE)v6 )
    goto LABEL_5;
  v2 = v3;
  if ( v7 )
    goto LABEL_7;
LABEL_16:
  if ( a1 + 8 != v2 )
  {
    if ( v4 == *(_DWORD *)(v2 + 32) )
    {
      if ( v4 <= 1 )
      {
        if ( *(_DWORD *)(a2 + 16) < *(_DWORD *)(v2 + 48) )
          return a1 + 8;
      }
      else
      {
        v13 = *(_QWORD *)(a2 + 40);
        v14 = *(_QWORD *)(v2 + 72);
        v15 = v14;
        if ( v13 <= v14 )
          v15 = *(_QWORD *)(a2 + 40);
        if ( v15 && (v16 = memcmp(*(const void **)(a2 + 32), *(const void **)(v2 + 64), v15)) != 0 )
        {
LABEL_26:
          if ( v16 < 0 )
            return a1 + 8;
        }
        else
        {
          v17 = v13 - v14;
          if ( v17 <= 0x7FFFFFFF )
          {
            if ( v17 >= (__int64)0xFFFFFFFF80000000LL )
            {
              v16 = v17;
              goto LABEL_26;
            }
            return a1 + 8;
          }
        }
      }
    }
    else if ( (signed int)v4 < *(_DWORD *)(v2 + 32) )
    {
      return a1 + 8;
    }
  }
  return v2;
}
